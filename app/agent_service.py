import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from langgraph.types import Command, interrupt

from app.config import config
from app.database import db
from app.models import AiChatRecord, WebSocketContentItem, WebSocketMessage, WebSocketMessageData
from app.utils import generate_chat_id, get_current_timestamp

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# 启动时加载静态资源
# ─────────────────────────────────────────────────────────────
_BASE = Path(__file__).parent.parent
_TABLE_LIST = (_BASE / "db_table_list.txt").read_text(encoding='utf-8') \
    if (_BASE / "db_table_list.txt").exists() else ""
_COL_MAP: dict = json.loads((_BASE / "db_columns.json").read_text(encoding='utf-8')) \
    if (_BASE / "db_columns.json").exists() else {}

# 表注释映射（表名 → 注释文字），从 db_table_list.txt 解析
_TABLE_COMMENT: Dict[str, str] = {}
for _line in _TABLE_LIST.splitlines():
    _line = _line.strip()
    if not _line or ' -- ' not in _line:
        continue
    _tbl_part, _comment_part = _line.split(' -- ', 1)
    _tbl_name = _tbl_part.strip().upper()
    # 注释取 "，含字段" 之前的部分，若无则取全部
    _comment = _comment_part.split('，含字段')[0].split(',含字段')[0].strip()
    if _tbl_name:
        _TABLE_COMMENT[_tbl_name] = _comment

# 领域→表映射配置（domain_tables.json）
_DOMAIN_TABLES_CFG: Dict[str, dict] = json.loads(
    (_BASE / "config" / "domain_tables.json").read_text(encoding='utf-8')
) if (_BASE / "config" / "domain_tables.json").exists() else {}
# 供意图识别用的领域说明（domain → name）
_DOMAIN_NAMES: Dict[str, str] = {k: v["name"] for k, v in _DOMAIN_TABLES_CFG.items()}

# 领域 QA 样例库（domain_qa.json）
_DOMAIN_QA: Dict[str, list] = json.loads(
    (_BASE / "config" / "domain_qa.json").read_text(encoding='utf-8')
) if (_BASE / "config" / "domain_qa.json").exists() else {}

_PERSON_REPORT_PROMPT = (_BASE / "prompt" / "person_report.txt").read_text(encoding='utf-8') \
    if (_BASE / "prompt" / "person_report.txt").exists() else ""

_PERSON_REPORT_TABLES = [
    (item["table"], item["label"], item["id_col"])
    for item in json.loads(
        (_BASE / "config" / "person_report.json").read_text(encoding='utf-8')
    )["tables"]
]

_BASE_PERSON_TABLES = [
    ("DIMENSYSTEM.PERSONNEL_PROFILE",  "全市公职人员数据",      "IDCARD",   "NAME"),
    ("DIMENSYSTEM.PUBLIC_INSTITUTION", "全市事业干部人员数据",   "ID_CARD",  "NAME"),
]
_BASE_TABLE_NAMES = {t[0].lower() for t in _BASE_PERSON_TABLES}

# ─────────────────────────────────────────────────────────────
# 权限配置
# ─────────────────────────────────────────────────────────────
# 有权访问人员数据的角色（person_report 意图 + nl2sql 查人员表）
_PERSON_DATA_ROLES = {"sld", "jwld", "admin", "swld"}

# ─────────────────────────────────────────────────────────────
# Prompt 模板
# ─────────────────────────────────────────────────────────────
_SELECT_TABLE_PROMPT = f"""你是一个数据库专家。根据用户的问题，从以下表列表中选出最相关的表，返回表的完整名称（schema.表名），每行一个，不要任何解释。

{_TABLE_LIST}

选表规则：
1. 优先选能单表回答问题的表，不要为了"更完整"而多选
2. 只有当问题明确需要多表数据时才选多张表（最多3张）
3. 不要选注释为空或与问题无关的表
4. Schema优先级：DIMENSYSTEM > DIMENDOMAIN > DIMENMODEL > DIMENUPLOAD
5. 如果完全找不到相关表，只返回：UNKNOWN

常用场景速查（优先按此规则选表）：
- 问"某人是谁/某职位/某单位的人员/名单/局长/科长" → DIMENSYSTEM.PERSONNEL_PROFILE
- 问事业单位干部、编制人员 → DIMENSYSTEM.PUBLIC_INSTITUTION
- 两张表都可能有时，优先选 DIMENSYSTEM.PERSONNEL_PROFILE

医保领域速查（含违规/预警/退还/转移等关键词时优先选以下表）：
- 问医保违规明细、触发规则、预警详情 → DIMENDOMAIN.YIBAO_VIOLATION_DETAIL_BAK
- 问医保退还记录、退款、退还金额 → DIMENDOMAIN.YIBAO_REFUND_RECORD_BAK
- 问异地就医违规、跨区域违规 → DIMENDOMAIN.YIBAO_CROSS_REGION_VIOLATION_BAK
- 问医保转移记录、参保转移、迁移 → DIMENDOMAIN.YIBAO_TRANSFER_RECORD_BAK

农村三资领域速查（含三资/集体资产/农村合同/扶贫资产/负债/现金/账务等关键词时优先选以下表）：
- 问农村三资基础数据、土地面积、总资产、总负债、所有者权益 → DIMENDOMAIN.NCJT_RURAL_THREE_ASSETS
- 问不支撑资产、资产未登记项目 → DIMENDOMAIN.NCJT_UNSUPPORTED_ASSETS
- 问扶贫资产未登记、应登记未登记数量 → DIMENDOMAIN.NCJT_UNREGISTERED_POVERTY_ASSETS
- 问月账未结清、账务未结、未结账数量 → DIMENDOMAIN.NCJT_UNCLOSED_MONTHLY_ACCOUNTS
- 问现金预警、现金超标、现金金额预警 → DIMENDOMAIN.NCJT_CASH_WARNING
- 问债务资产预警、负债率预警、资产负债比 → DIMENDOMAIN.NCJT_DEBT_ASSET_WARNING
- 问资产出租合同、有租无合同、合同核查 → DIMENDOMAIN.NCJT_PROPERTY_RENTAL_CONTRACT
- 问资产交易、平台交易、有合同无平台、有平台无合同 → DIMENDOMAIN.NCJT_PROPERTY_TRANSACTION
- 问长期合同预警、合同期限过长、甲方乙方合同信息 → DIMENDOMAIN.NCJT_LONG_TERM_CONTRACT_WARNING
- 问到期合同预警、合同到期未续签 → DIMENDOMAIN.NCJT_EXPIRED_CONTRACT_WARNING
- 问合同整改问题、整改状态、问题描述、整改措施 → DIMENDOMAIN.NCJT_CONTRACT_RECTIFICATION_ISSUE
- 问合同整改统计、核查总数、问题合同数、已整改数 → DIMENDOMAIN.NCJT_CONTRACT_RECTIFICATION_STAT

高标准农田领域速查（含高标准农田/农田建设/招投标/验收/管护/选址等关键词时优先选以下表）：
- 问高标准农田基本情况、项目列表、建设规模、计划投资、实际结算 → DIMENDOMAIN.HS_FARMLAND_BASIC_INFO
- 问标段划分、疑似不合理标段、采购方式分析 → DIMENDOMAIN.HS_AGRICULTURAL_PROJECTS
- 问招投标详情、中标单位、施工单位、设计单位、监理单位、预警等级 → DIMENDOMAIN.HS_BIDDING_DETAIL_LIST
- 问资金使用、资金迟拨、审减比例过高、支付比例 → DIMENDOMAIN.HS_FUND_USAGE_MANAGEMENT
- 问竣工验收、验收问题、整改不到位、验收超时、结算审核超时 → DIMENDOMAIN.HS_GBT_ACCEPTANCE
- 问建后管护、管护资金拨付、管护不到位 → DIMENDOMAIN.HS_GBT_MAINTENANCE
- 问不合规选址、25度耕地、生态保护红线、水源地、自然保护地 → DIMENDOMAIN.HS_NON_COMPLIANT_SITE_SELECTION

六霸领域速查（含六霸/村霸/乡霸/沙霸/街霸/市霸/矿霸/线索/战果/民警等关键词时优先选以下表）：
- 问六霸战果主体信息、案件概况、侦查终结、在逃人员 → DIMENDOMAIN.T_SIX_BULLIES_RESULT
- 问战果下属案件、案件类型、案件状态 → DIMENDOMAIN.T_SIX_BULLIES_RESULT_CASE
- 问办案民警、主办民警、协办民警、警号、职务职级 → DIMENDOMAIN.T_SIX_BULLIES_RESULT_OFFICER
- 问六霸案件清单、案由、承办人、警情编号 → DIMENDOMAIN.T_SIX_BULLIES_CASE
- 问六霸线索、省厅转办线索、自行摸排线索、核查进展 → DIMENDOMAIN.T_SIX_BULLIES_CLUE
- 问六霸类犯罪介绍、各类型说明 → DIMENDOMAIN.T_SIX_BULLIES_INTRO

养老服务领域速查（含养老/机构/护理/床位/高龄津贴/供餐/适老化/改造/居家养老等关键词时优先选以下表）：
- 问养老机构基本情况、床位数量、机构类型、负责人 → DIMENDOMAIN.ELDERLY_CARE_ORG
- 问养老机构入住人员、入住老人、能力评估 → DIMENDOMAIN.ELDERLY_CARE_RESIDENT
- 问养老机构工作人员、护理员、证件等级 → DIMENDOMAIN.ELDERLY_CARE_STAFF
- 问镇村养老服务设施、综合养老服务中心、居家养老服务中心 → DIMENDOMAIN.ELDERLY_SERVICE_FACILITY
- 问双空白整改、设施空白整改情况、责任人 → DIMENDOMAIN.ELDERLY_DOUBLE_BLANK_RECTIFY
- 问养老机构排查问题、问题整改台账、整改措施 → DIMENDOMAIN.ELDERLY_ORG_PROBLEM_RECTIFY
- 问供餐情况、餐标、食品留样、陪餐制度、食品经营许可证 → DIMENDOMAIN.ELDERLY_MEAL_SUPPLY_INFO
- 问食品安全预警、供餐预警信息 → DIMENDOMAIN.ELDERLY_MEAL_WARNING
- 问违规发放高龄津贴、死亡未停发、户籍迁移未停发、追缴情况 → DIMENDOMAIN.ELDERLY_ALLOWANCE_VIOLATION
- 问养老服务资金项目、资金支出率、未支出资金原因 → DIMENDOMAIN.ELDERLY_FUND_PROJECT_DETAIL
- 问移交纪检监察线索、民政部门移交问题线索 → DIMENDOMAIN.ELDERLY_TRANSFER_CLUE
- 问适老化改造明细、改造地址、改造物品、改造金额 → DIMENDOMAIN.ELDERLY_AGING_HOME_RENOVATION
- 问适老化改造预警、预警问题整改明细 → DIMENDOMAIN.ELDERLY_AGING_RENOVATION_WARNING

乡镇财政领域速查（含乡镇财政/账户/镇街/预警/补贴/项目穿透/偏离度等关键词时优先选以下表）：
- 问乡镇账户汇总、账户类型、开户银行、账户状态 → DIMENDOMAIN.T_FINANCE_ACCOUNT_SUMMARY
- 问镇街银行账户清单、账户预警数量、镇长信息 → DIMENDOMAIN.T_FINANCE_TOWN_ACCOUNT
- 问账户预警信息、预警事项、支出凭证 → DIMENDOMAIN.T_FINANCE_WARNING
- 问四类补贴、耕地地力保护补贴、海洋渔业补贴、休渔补贴发放 → DIMENDOMAIN.T_FINANCE_SUBSIDY
- 问项目信息明细、基建项目、建设单位、项目预警 → DIMENDOMAIN.T_FINANCE_PROJECT_DETAIL
- 问项目支付情况、发包人合同支出、支出进度 → DIMENDOMAIN.T_FINANCE_PROJECT_PAYMENT
- 问发包人资金支付明细、付款凭证、收款人信息 → DIMENDOMAIN.T_FINANCE_PROJECT_EMPLOYER
- 问承包人资金支付明细、承包人合同、分包人 → DIMENDOMAIN.T_FINANCE_PROJECT_CONTRACTOR
- 问项目穿透式预警、监控规则、违规问题分类、整改方式 → DIMENDOMAIN.T_FINANCE_PROJECT_WARNING
- 问县市区预算执行偏离度汇总 → DIMENDOMAIN.T_FINANCE_COUNTY_DEVIATION
- 问各镇街预算执行偏离度明细、偏离度预警 → DIMENDOMAIN.T_FINANCE_TOWN_DEVIATION

信访领域速查（含信访/办结率/信访问题/集中治理等关键词时优先选以下表）：
- 问信访办结率、月度办结率、预警状态 → DIMENDOMAIN.PETITION_COMPLETION_RATE_MONTHLY
- 问信访问题清单、信访人、诉求、化解进展、交办层级 → DIMENDOMAIN.PETITION_GOVERNANCE_WORK

违规执法领域速查（含违规执法/乱罚款/乱查封/乱检查/乱收费/趋利性执法/异地执法/涉企/行政处罚/积案/冻结账户等关键词时优先选以下表）：
- 问查纠突出问题清单、问题性质、整改效果 → DIMENDOMAIN.WGZF_PROBLEM
- 问涉企行政处罚案件、罚款金额、立案日期 → DIMENDOMAIN.WGZF_PENALTY
- 问涉企行政检查案件、检查类型、检查结果 → DIMENDOMAIN.WGZF_INSPECT
- 问行政机关败诉案件、败诉原因、审判机关 → DIMENDOMAIN.WGZF_LOSE_CASE
- 问行政全量案件数据 → DIMENDOMAIN.WGZF_TOTAL_CASE
- 问清理涉企积案、积案原因 → DIMENDOMAIN.WGZF_BACKLOG
- 问变更或解除冻结账户 → DIMENDOMAIN.WGZF_FROZEN_ACCT
- 问释放资金 → DIMENDOMAIN.WGZF_RELEASE_FUND
- 问返还违规扣押资金 → DIMENDOMAIN.WGZF_RETURN_FUND

三张清单/哨兵领域速查（含三张清单/廉政风险点/业务风险点/群众反映集中点/监督预警/纪检组等关键词时优先选以下表）：
- 问预警工单、12345预警、派发状态、办理状态 → DIMENDOMAIN.SUPERVISE_WARNING
- 问廉政风险点清单、廉政风险表现、监督规则 → DIMENDOMAIN.SUPERVISE_LIST_INTEGRITY
- 问业务风险点清单、业务风险表现 → DIMENDOMAIN.SUPERVISE_LIST_BUSINESS
- 问群众反映集中点清单、群众诉求集中点 → DIMENDOMAIN.SUPERVISE_LIST_PUBLIC

教育领域速查（含教育/学生/收费/补贴/应收/应免/违纪违法等关键词时优先选以下表）：
- 问学生收费信息、应收未收、应免未免、学校负责人 → DIMENDOMAIN.STUDENT_SUBSIDY_INFO
- 问教育成效分析、主动交代金额、违纪违法类型 → DIMENDOMAIN.EDUCATION_EFFECT_ANALYSIS

国有资产领域速查（含国有资产/出租/闲置/土地/林地/回避/机关资产等关键词时优先选以下表）：
- 问干部人员全量、人员职务 → DIMENDOMAIN.GZ_STAFF_ALL
- 问空编人员、空编原因 → DIMENDOMAIN.GZ_IDLE_STAFF
- 问回避关系、利益关联 → DIMENDOMAIN.GZ_AVOID
- 问资产出租台账、承租人、租金、合同期限 → DIMENDOMAIN.GZ_RENT_LEDGER
- 问出租期限过长预警 → DIMENDOMAIN.GZ_RENT_OVERLONG
- 问低价出租预警、价格比率 → DIMENDOMAIN.GZ_RENT_LOWPRICE
- 问闲置资产、资产闲置原因 → DIMENDOMAIN.GZ_IDLE_ASSET
- 问土地储备、土地用途 → DIMENDOMAIN.GZ_LAND_RESERVE
- 问林地资源台账 → DIMENDOMAIN.GZ_FOREST_RES
- 问林地问题台账、林地整改 → DIMENDOMAIN.GZ_FOREST_ISSUE
- 问机关资产台账、资产编码、使用状态 → DIMENDOMAIN.GZ_ORGAN_ASSET

网络餐饮领域速查（含网络餐饮/骑手/平台/举报/投诉/执法问责/金地湾等关键词时优先选以下表）：
- 问食品举报投诉、被投诉企业、诉求内容 → DIMENDOMAIN.FOOD_REPORT_COMPLAINT
- 问骑手信息、健康证、网点管理员 → DIMENDOMAIN.PLATFORM_RIDER_DETAIL
- 问餐饮监管执法记录、典型问题、违规处置 → DIMENDOMAIN.CATERING_ENFORCEMENT_RECORD
- 问网络平台举报投诉 → DIMENDOMAIN.NETWORK_PLATFORM_REPORT_COMPLAINT
- 问金地湾网络餐饮投诉 → DIMENDOMAIN.JINDIWAN_NETWORK_CATERING_COMPLAINT
- 问金地湾执法问责 → DIMENDOMAIN.JINDIWAN_ENFORCEMENT_ACCOUNTABILITY

治理拖欠农民工工资领域速查（含拖欠工资/欠薪/农民工/工资保证金/建设单位/施工单位等关键词时优先选以下表）：
- 问欠薪项目总体情况、建设单位、施工单位、涉及金额、整改情况 → DIMENDOMAIN.WAGE_PROJECT_OVERVIEW
- 问欠薪原因分析、原因分类、解决方式 → DIMENDOMAIN.WAGE_ARREARS_REASON
- 问制度落实检查情况、实名制、工资保证金、工资专用账户 → DIMENDOMAIN.WAGE_SYSTEM_CHECK
- 问欠薪案件处置、追缴金额、解决人数 → DIMENDOMAIN.WAGE_CASE_DISPOSAL

肉制品治理领域速查（含肉制品/假劣/屠宰/检疫/开证/抽检/涉肉/犯罪等关键词时优先选以下表）：
- 问养殖开证信息、检疫证号、货主、开证人 → DIMENDOMAIN.HS_BREEDING_CERT_INFO
- 问屠宰场检疫数量、用水用电、异常预警 → DIMENDOMAIN.HS_SLAUGHTER_QUARANTINE
- 问屠宰投诉举报 → DIMENDOMAIN.HS_SLAUGHTER_COMPLAINT
- 问屠宰行政执法、罚款金额、移送公安 → DIMENDOMAIN.HS_SLAUGHTER_ADMIN_LAW
- 问抽检合格清单、抽检单位、抽检人员 → DIMENDOMAIN.HS_SAMPLING_QUALIFIED
- 问抽检不合格清单、问题类型、溯源、责任人 → DIMENDOMAIN.HS_SAMPLING_UNQUALIFIED
- 问投诉信息、登记编号、承办机构、处理结果 → DIMENDOMAIN.HS_COMPLAINT_INFO
- 问举报信息 → DIMENDOMAIN.HS_REPORT_INFO
- 问监管排查问题清单、经营主体、整改落实 → DIMENDOMAIN.HS_SUPERVISION_PROBLEM
- 问行政执法案件、违法行为、溯源、移送公安 → DIMENDOMAIN.HS_ADMIN_LAW_CASE
- 问涉肉刑事案件、查获数量、涉案金额、犯罪嫌疑人 → DIMENDOMAIN.HS_CRIMINAL_CASE
"""

_GEN_SQL_PROMPT_TPL = """你是一个达梦数据库SQL专家。根据用户问题和以下表结构，生成对应的SQL查询语句。

{schema_info}

要求：
1. 只返回SQL语句，不要任何解释文字
2. SQL语句以分号结尾
3. 表名必须带schema前缀（如 DIMENSYSTEM.PERSONNEL_PROFILE）
4. 默认加 LIMIT 100
5. 优先用单表查询，只有字段确实分布在多张表时才 JOIN
6. JOIN 前确认关联字段在两张表中都存在且有实际数据
7. 【严格禁止】只能使用上方表结构中列出的表，禁止使用任何未在上方出现的表名

【字段语义拆分规则 — 必须严格遵守】
8. 用户问"某单位某职位是谁"时，必须将条件拆分到对应字段，禁止把整个短语放在 NAME 字段：
   - 单位名称（如：汕尾市红十字会、财政局、教育局）→ UNIT 字段
   - 职务/职级（如：局长、副局长、主任、副主任、科长、书记）→ CURRENTPOSITION 或 POSITION 字段
   - 人名（如：张三、李四）→ NAME 字段
   示例：
     问：汕尾市红十字会副局长是谁
     错误：WHERE NAME = '汕尾市红十字会副局长'
     正确：WHERE UNIT = '汕尾市红十字会' AND CURRENTPOSITION = '副局长'

9. 当问题中包含"局长/主任/书记/科长/院长"等职务词时，优先用职务字段过滤，而非 NAME 字段。

10. 单位名称可能是简称，使用 LIKE '%关键词%' 做模糊匹配更稳健：
    示例：WHERE UNIT LIKE '%红十字会%' AND CURRENTPOSITION LIKE '%副局长%'
    SELECT "医药机构", "违规金额", "触发规则" FROM DIMENDOMAIN.YIBAO_VIOLATION_DETAIL_BAK LIMIT 100;
11. 英文列名的医保表（YIBAO_CROSS_REGION_VIOLATION_BAK）无需双引号。
"""


# ─────────────────────────────────────────────────────────────
# 人员报告字段描述（启动时构建）
# ─────────────────────────────────────────────────────────────
def _build_person_report_schema() -> Dict[str, str]:
    """为 person_report.json 中每张表构建字段描述，供 AI 参考。"""
    schema_map: Dict[str, str] = {}
    target_tables = {item["table"].upper() for item in
                     json.loads((_BASE / "config" / "person_report.json").read_text(encoding='utf-8'))["tables"]}

    compact_path = _BASE / "db_schema_compact.txt"
    if compact_path.exists():
        text = compact_path.read_text(encoding='utf-8')
        blocks = re.split(r'(?=^表: )', text, flags=re.MULTILINE)
        for block in blocks:
            m = re.match(r'^表: (\S+)', block)
            if not m:
                continue
            tbl = m.group(1).upper()
            if tbl in target_tables:
                schema_map[tbl.lower()] = block.strip()

    missing = [t for t in target_tables if t.lower() not in schema_map]
    if missing:
        try:
            for full_name in missing:
                parts = full_name.split(".")
                if len(parts) != 2:
                    continue
                owner, tbl_name = parts[0], parts[1]
                rows = db.execute_query(
                    "SELECT COLUMN_NAME, DATA_TYPE FROM ALL_TAB_COLUMNS "
                    "WHERE OWNER = ? AND TABLE_NAME = ? ORDER BY COLUMN_ID",
                    (owner, tbl_name)
                )
                if rows:
                    lines = [f"表: {full_name}"]
                    lines += [f"  {r['COLUMN_NAME']} {r['DATA_TYPE']}" for r in rows]
                    schema_map[full_name.lower()] = "\n".join(lines)
        except Exception as e:
            logger.warning(f"[人员报告] 查询 ALL_TAB_COLUMNS 失败: {e}")

    logger.info(f"[人员报告] 字段描述加载完成，共 {len(schema_map)} 张表")
    return schema_map


_PERSON_REPORT_SCHEMA: Dict[str, str] = _build_person_report_schema()

# ─────────────────────────────────────────────────────────────
# 医院别称配置
# ─────────────────────────────────────────────────────────────
_HOSPITAL_ALIASES: dict = json.loads(
    (_BASE / "config" / "hospital_aliases.json").read_text(encoding='utf-8')
) if (_BASE / "config" / "hospital_aliases.json").exists() else {"hospitals": []}


def _find_hospitals_by_alias(target: str) -> list:
    """查找医院别称。返回所有匹配的医院配置列表；target 已是全称则返回空列表。"""
    target = target.strip()
    results = []
    for hospital in _HOSPITAL_ALIASES.get("hospitals", []):
        full_name = hospital.get("full_name", "")
        if target == full_name:
            return []  # 已经是全称，无需处理
        if target in hospital.get("aliases", []):
            results.append(hospital)
    return results


# ─────────────────────────────────────────────────────────────
# 通用 domain 报告配置（17 个领域统一管理）
# ─────────────────────────────────────────────────────────────
_DOMAIN_CONFIG_MAP: Dict[str, str] = {
    "med_fund_mgmt":        "med_fund_mgmt_report.json",
    "edu_profit":           "edu_profit_report.json",
    "elderly_care":         "elderly_care_report.json",
    "rural_assets":         "rural_assets_report.json",
    "state_assets":         "state_assets_report.json",
    "township_finance":     "township_finance_report.json",
    "farmland":             "farmland_report.json",
    "six_bullies":          "six_bullies_report.json",
    "petition":             "petition_report.json",
    "illegal_enforcement":  "illegal_enforcement_report.json",
    "supervise":            "supervise_report.json",
    "edu_subsidy":          "edu_profit_report.json",
    "online_catering":      "online_catering_report.json",
    "wage_arrears":         "wage_arrears_report.json",
    "meat_products":        "meat_products_report.json",
}

_DOMAIN_REPORT_CONFIGS: Dict[str, dict] = {}
# key: "domain" 或 "domain_subtype"（医保多子类型用 "med_fund_mgmt_area" 等）
_DOMAIN_REPORT_PROMPTS: Dict[str, str] = {}

for _domain, _cfg_file in _DOMAIN_CONFIG_MAP.items():
    _cfg_path = _BASE / "config" / _cfg_file
    if not _cfg_path.exists():
        print(f"[domain_report] 配置文件不存在，跳过: {_cfg_path}", flush=True)
        continue
    _dcfg = json.loads(_cfg_path.read_text(encoding='utf-8'))
    _DOMAIN_REPORT_CONFIGS[_domain] = _dcfg
    for _subtype, _subcfg in _dcfg.items():
        _prompt_file = _subcfg.get("prompt", "")
        if not _prompt_file:
            continue
        _dp = _BASE / "prompt" / _prompt_file
        # 单子类型（default）直接用 domain 作 key；多子类型用 "domain_subtype"
        _pkey = _domain if _subtype == "default" else f"{_domain}_{_subtype}"
        _DOMAIN_REPORT_PROMPTS[_pkey] = _dp.read_text(encoding='utf-8') if _dp.exists() else ""



# ─────────────────────────────────────────────────────────────
# LangGraph State
# ─────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    session_id: str
    ai_model: str
    content: str        # 当前用户输入
    record_id: str      # 前端消息归位ID，透传回所有推送消息
    user_roles: list    # 前端传入的用户角色列表，用于权限控制
    history: list       # 最近 N 轮对话历史 [{"role": "user/assistant", "content": "..."}]
    intent: str         # 意图分类结果
    name: str           # 提取的人名
    id_card: str        # 确认后的身份证号
    candidates: list    # 重名候选列表
    base_data: dict     # 基础表查询结果
    sql: str            # 生成的 SQL
    query_results: list # SQL 执行结果
    report: str         # 最终报告文本
    # 通用 domain 报告
    domain: str                # 当前请求的 domain（从 extend 传入）
    domain_subtype: str        # 报告子类型（医保用 area/hospital/department，其他用 default）
    domain_target: str         # 目标名称（医保医院/科室名称，其他为空）
    domain_report_schema: str  # 供 AI 生成 SQL 的 schema 描述
    domain_report_sqls: list   # AI 生成的聚合 SQL 列表
    domain_report_answer: str  # 主数据聚合结果
    domain_report_extra: str   # 补充数据聚合结果


# ─────────────────────────────────────────────────────────────
# AgentService
# ─────────────────────────────────────────────────────────────
class AgentService:

    def __init__(self):
        self.llm = ChatOpenAI(
            api_key=config.ai.get('api_key'),
            base_url=config.ai.get('base_url'),
            model=config.ai.get('model'),
            temperature=0,
            timeout=60,
            max_retries=1,
        )
        fast_model = config.ai.get('fast_model', config.ai.get('model'))
        self.fast_llm = ChatOpenAI(
            api_key=config.ai.get('api_key'),
            base_url=config.ai.get('base_url'),
            model=fast_model,
            temperature=0,
            timeout=60,
            max_retries=1,
        )
        # WebSocket 注册表（不进 graph state，避免序列化问题）
        self._ws_registry: Dict[str, Any] = {}
        # 已中止的会话集合，下次发消息时强制新建图而不是 resume
        self._aborted_sessions: set = set()
        self.checkpointer = MemorySaver()
        self.graph = self._build_graph()

    # ── 聊天记录 ──────────────────────────────────────────────

    async def get_chat_records(self, session_id: str, cursor: int, limit: int) -> List[AiChatRecord]:
        try:
            records_data = db.get_chat_records(session_id, limit, cursor)
            records = []
            for data in records_data:
                record = AiChatRecord(
                    chat_id=data.get('CHAT_ID'),
                    user_id=data.get('USER_ID', ''),
                    session_id=data.get('SESSION_ID'),
                    ai_model=data.get('AI_MODEL'),
                    role=data.get('ROLE'),
                    content=data.get('CONTENT'),
                    user_prompt=data.get('USER_PROMPT'),
                    system_prompt=data.get('SYSTEM_PROMPT'),
                    create_time=data.get('CREATE_TIME'),
                    prompt_tokens=data.get('PROMPT_TOKENS', 0),
                    completion_tokens=data.get('COMPLETION_TOKENS', 0),
                    total_tokens=data.get('TOTAL_TOKENS', 0)
                )
                records.append(record)
            return records
        except Exception as e:
            logger.error(f"获取聊天记录失败: {e}")
            return []

    async def save_chat_record(self, record: AiChatRecord):
        logger.info(f"保存聊天记录: SessionID={record.session_id}, Role={record.role}")
        try:
            db.insert_chat_record({
                'chat_id': record.chat_id,
                'user_id': record.user_id,
                'session_id': record.session_id,
                'ai_model': record.ai_model,
                'role': record.role,
                'content': record.content,
                'user_prompt': record.user_prompt,
                'system_prompt': record.system_prompt,
                'create_time': record.create_time,
                'prompt_tokens': record.prompt_tokens,
                'completion_tokens': record.completion_tokens,
                'total_tokens': record.total_tokens,
            })
        except Exception as e:
            logger.error(f"保存聊天记录失败: {e}")

    # ── 工具方法 ──────────────────────────────────────────────

    @staticmethod
    def _extract_sql(text: str) -> str:
        match = re.search(r'```(?:sql)?\s*([\s\S]+?)```', text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return text.strip()

    @staticmethod
    def _mask_idcard(idcard: str) -> str:
        if len(idcard) >= 10:
            return idcard[:6] + "****" + idcard[-4:]
        return idcard[:2] + "****" if len(idcard) > 2 else "****"

    @staticmethod
    def _build_candidate_message(candidates: list) -> str:
        lines = [f"找到 {len(candidates)} 位同名人员，请回复序号选择："]
        for i, c in enumerate(candidates, 1):
            unit = c.get("unit", "")
            pos = c.get("position", "")
            masked = AgentService._mask_idcard(c["idcard"])
            lines.append(f"{i}. {c['name']} — {unit}{'—' + pos if pos else ''}（身份证：{masked}）")
        return "\n".join(lines)

    def _query_person_data(self, id_card: str, skip_tables: Optional[set] = None) -> dict:
        """按身份证号查询 person_report.json 中的表，skip_tables 中的表跳过（小写表名）"""
        skip_tables = skip_tables or set()
        results = {}
        for table, label, id_col in _PERSON_REPORT_TABLES:
            if table.lower() in skip_tables:
                continue
            try:
                rows = db.execute_query(
                    f"SELECT * FROM {table} WHERE {id_col} = ? LIMIT 50",
                    (id_card,)
                )
                if rows:
                    results[label] = rows
                    logger.info(f"  {label}: {len(rows)} 条")
                else:
                    logger.info(f"  {label}: 无数据")
            except Exception as e:
                logger.warning(f"  {label} 查询失败: {e}")
        return results

    def _format_results(self, results: list) -> str:
        if not results:
            return "无数据"
        columns = list(results[0].keys())
        lines = [
            "| " + " | ".join(columns) + " |",
            "|" + "|".join(["---" for _ in columns]) + "|",
        ]
        for row in results:
            values = [str(row.get(col, '')) for col in columns]
            lines.append("| " + " | ".join(values) + " |")
        return "\n".join(lines)

    # ── WebSocket 推送 ────────────────────────────────────────

    async def _send_progress(self, websocket, ai_model: str, text: str, record_id: str = ""):
        if not websocket:
            return
        msg = WebSocketMessage(
            type="progress",
            data=WebSocketMessageData(
                role="assistant",
                aiModel=ai_model,
                content=[WebSocketContentItem(type="text", content=text, record_id=record_id)]
            ),
            done=False
        )
        await websocket.send_json(msg.dict())

    async def _stream_response(self, websocket, response: str, ai_model: str, record_id: str = ""):
        chunk_size = 20
        for i in range(0, len(response), chunk_size):
            chunk = response[i:i + chunk_size]
            done = (i + chunk_size) >= len(response)
            message = WebSocketMessage(
                type="aiMessage",
                data=WebSocketMessageData(
                    role="assistant",
                    aiModel=ai_model,
                    content=[WebSocketContentItem(type="text", content=chunk, record_id=record_id)]
                ),
                done=done
            )
            await websocket.send_json(message.dict())
            await asyncio.sleep(0.02)

    async def _stream_ai_to_ws(self, prompt: str, websocket, ai_model: str,
                               temperature: float = 0.3, record_id: str = "",
                               history: list = None) -> str:
        """流式调用 AI，边生成边推送到 WebSocket，返回完整文本（用于存库）。
        history: [{"role": "user"/"assistant", "content": "..."}] 多轮历史消息
        """
        from langchain_core.messages import AIMessage
        llm = self.llm.bind(temperature=temperature)
        # 构建多轮消息列表
        messages = []
        for msg in (history or []):
            if msg["role"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"] == "assistant":
                messages.append(AIMessage(content=msg["content"]))
        messages.append(HumanMessage(content=prompt))

        full_text = ""
        async for chunk in llm.astream(messages):
            delta = chunk.content or ""
            if delta:
                full_text += delta
                if websocket:
                    await websocket.send_json(WebSocketMessage(
                        type="aiMessage",
                        data=WebSocketMessageData(
                            role="assistant",
                            aiModel=ai_model,
                            content=[WebSocketContentItem(type="text", content=delta, record_id=record_id)]
                        ),
                        done=False
                    ).dict())
        if websocket:
            await websocket.send_json(WebSocketMessage(
                type="aiMessage",
                data=WebSocketMessageData(
                    role="assistant",
                    aiModel=ai_model,
                    content=[WebSocketContentItem(type="text", content="", record_id=record_id)]
                ),
                done=True
            ).dict())
        logger.info(f"[流式输出] 完成，共 {len(full_text)} 字符")
        return full_text

    # ── LangGraph 节点 ────────────────────────────────────────

    async def _node_classify_intent(self, state: AgentState) -> dict:
        logger.info(">>> 节点: classify_intent（意图识别）")

        # 前端已明确传入 domain，直接走 nl2sql（领域限定选表），跳过意图识别
        frontend_domain = state.get("domain", "").strip()
        if frontend_domain and frontend_domain in _DOMAIN_TABLES_CFG:
            logger.info(f"[意图识别] 前端传入 domain={frontend_domain}，直接走 nl2sql（领域限定）")
            return {"intent": "nl2sql", "domain": frontend_domain}

        # 构建领域列表供 LLM 参考
        domain_list = "\n".join(
            f"  · {k:<28} — {v}" for k, v in _DOMAIN_NAMES.items()
        )

        prompt = (
            "你是一个意图分类器。根据用户输入，判断属于以下哪种意图，"
            "只返回对应的英文标识，不要任何解释：\n\n"
            "- person_report：用户明确要求生成/输出/查看某个自然人的综合档案或信息报告\n"
            "  触发词：'的报告'、'的档案'、'的信息报告'、'个人情况'、'综合报告'\n"
            "  【必须同时满足】：① 包含明确自然人姓名（2-4个汉字人名，非机构/单位/医院/学校等组织名称）② 包含报告/档案相关词\n"
            "  示例：张三的人员信息报告、生成李四的档案、查王五的个人情况\n"
            "  不触发：'张三在哪工作'、'查一下财政局局长'（这些是 nl2sql）\n"
            "  不触发：'中山大学孙逸仙纪念医院的报告'、'财政局的报告'、'某医院出一份报告'（机构/单位/医院名称不是自然人姓名）\n\n"
            "- domain_report:<domain>：用户要求针对某个机构、区域或专项领域生成分析报告\n"
            "  触发条件：包含'出一份报告'、'生成报告'、'分析报告'、'专项报告'、'出报告'等词，且对象是机构/区域/领域而非自然人\n"
            "  示例：针对某医院出一份报告、生成医保分析报告、出一份养老服务报告、针对某区域的乡镇财政报告\n"
            "  在冒号后附上对应 domain 值，领域关键词速查同 nl2sql 规则\n"
            f"  可选 domain 值：\n{domain_list}\n\n"
            "- nl2sql:<domain>：查询数据库中的结构化数据\n"
            "  包括：统计类、列表类、按条件查询、询问某领域的具体数据\n"
            "  示例：医保违规问题有哪些、鼓楼区预警金额是多少、财政局有哪些科长\n"
            "  如果问题明确属于某个专项领域，在冒号后附上对应 domain 值；\n"
            "  如果无法判断领域或属于通用查询，直接返回 nl2sql（不带冒号）\n"
            "  领域关键词速查（遇到以下关键词时优先附上对应 domain）：\n"
            "    · 医保/医院/医疗机构/药店/违规就医/报销/参保/民营医院/公立医院/科室/骗保/病院 → med_fund_mgmt\n"
            "    · 养老/养老机构/高龄津贴/护理员/床位/居家养老/适老化/供餐 → elderly_care\n"
            "    · 农村三资/集体资产/农村合同/扶贫资产/月账/现金预警/债务资产 → rural_assets\n"
            "    · 国有资产/机关资产/出租/闲置/土地储备/林地/回避/空编 → state_assets\n"
            "    · 乡镇财政/镇街账户/四类补贴/项目穿透/偏离度/乡镇预算 → township_finance\n"
            "    · 高标准农田/农田建设/招投标/验收/管护/选址/标段/亩 → farmland\n"
            "    · 六霸/村霸/乡霸/沙霸/街霸/矿霸/线索/战果/民警 → six_bullies\n"
            "    · 信访/办结率/信访问题/集中治理 → petition\n"
            "    · 异地执法/违规执法/乱罚款/乱查封/乱检查/乱收费/趋利性执法/行政处罚/积案 → illegal_enforcement\n"
            "    · 三张清单/廉政风险/业务风险/群众反映/监督预警/哨兵 → supervise\n"
            "    · 学生/教育/学生收费/补贴/应收未收/应免未免/校服/教辅 → edu_subsidy\n"
            "    · 网络餐饮/骑手/平台/食品举报/投诉/金地湾 → online_catering\n"
            "    · 拖欠工资/欠薪/农民工/工资保证金 → wage_arrears\n"
            "    · 肉制品/屠宰/检疫/抽检/涉肉/假劣 → meat_products\n"
            f"  可选 domain 值：\n{domain_list}\n\n"
            "- knowledge：询问概念定义、政策法规、操作方法等知识性问题\n"
            "  （如：什么是三公经费、如何申请行政许可）\n\n"
            "- normal：其他普通对话\n\n"
            "以下是本次会话的历史对话（最近几轮，供参考上下文）：\n"
            "{history_context}"
            "当前用户输入：{content}"
        ).format(
            history_context="".join(
                f"{'用户' if h['role'] == 'user' else '助手'}：{h['content']}\n"
                for h in (state.get("history") or [])
            ) or "（无历史）\n",
            content=state["content"]
        )
        intent = "nl2sql"
        detected_domain = ""
        try:
            from langchain_core.messages import AIMessage
            history_msgs = []
            for h in (state.get("history") or []):
                if h["role"] == "user":
                    history_msgs.append(HumanMessage(content=h["content"]))
                elif h["role"] == "assistant":
                    history_msgs.append(AIMessage(content=h["content"]))
            resp = await asyncio.to_thread(
                self.fast_llm.invoke,
                history_msgs + [HumanMessage(content=prompt)]
            )
            raw = resp.content.strip().lower()
            if raw.startswith("domain_report:"):
                parsed_domain = raw.split(":", 1)[1].strip()
                intent = "domain_report"
                if parsed_domain in _DOMAIN_TABLES_CFG:
                    detected_domain = parsed_domain
                    logger.info(f"[意图识别] domain_report，识别到 domain={detected_domain}")
                else:
                    logger.info(f"[意图识别] domain_report，domain '{parsed_domain}' 不在配置中，domain 置空")
            elif raw.startswith("nl2sql:"):
                parsed_domain = raw.split(":", 1)[1].strip()
                intent = "nl2sql"
                if parsed_domain in _DOMAIN_TABLES_CFG:
                    detected_domain = parsed_domain
                    logger.info(f"[意图识别] nl2sql，识别到 domain={detected_domain}")
                else:
                    logger.info(f"[意图识别] nl2sql，domain '{parsed_domain}' 不在配置中，按全库选表")
            elif raw == "nl2sql":
                intent = "nl2sql"
            elif raw == "person_report":
                intent = "person_report"
            elif raw == "knowledge":
                intent = "knowledge"
            elif raw == "normal":
                intent = "normal"
            else:
                logger.warning(f"[意图识别] 模型返回未知意图 '{raw}'，降级为 nl2sql")
                intent = "nl2sql"
        except Exception as e:
            logger.error(f"[意图识别] AI 调用失败: {e}，降级为 nl2sql")
            intent = "nl2sql"
        logger.info(f"[意图识别] intent={intent} domain={detected_domain or '(空)'}")
        return {"intent": intent, "domain": detected_domain}

    async def _node_extract_name(self, state: AgentState) -> dict:
        logger.info(">>> 节点: extract_name（提取人名）")
        ws = self._ws_registry.get(state["session_id"])
        # 权限检查：只有指定角色才能访问人员数据
        user_roles = set(state.get("user_roles") or [])
        if not user_roles & _PERSON_DATA_ROLES:
            logger.warning(f"[权限拦截] person_report 被拒绝 | SessionID={state['session_id']} Roles={user_roles}")
            if ws:
                await self._stream_response(ws, "抱歉，您没有权限查询人员信息。", state["ai_model"], record_id=state.get("record_id", ""))
            return {"candidates": [], "name": ""}
        logger.info(f"[权限通过] person_report | SessionID={state['session_id']} Roles={user_roles}")
        if ws:
            await self._send_progress(ws, state["ai_model"], "🔍 正在识别人员信息...", record_id=state.get("record_id", ""))
        try:
            resp = await asyncio.to_thread(
                self.fast_llm.invoke,
                [HumanMessage(content=(
                    "从以下文本中提取被查询人的姓名。"
                    "注意：'人员信息报告'、'人员报告'、'信息报告'、'综合报告'等是固定词组，不是人名的一部分。"
                    "只返回人名，没有则返回空字符串。\n"
                    f"文本：{state['content']}"
                ))]
            )
            name = resp.content.strip()
        except Exception as e:
            logger.error(f"[人员报告] 提取人名失败: {e}")
            name = ""
        logger.info(f"[人员报告] 目标人员: {name}")
        return {"name": name}

    async def _node_lookup_person(self, state: AgentState) -> dict:
        logger.info(">>> 节点: lookup_person（查询基础人员表）")
        ws = self._ws_registry.get(state["session_id"])
        name = state["name"]
        if not name:
            if ws:
                await self._stream_response(ws, "请告诉我要查询哪位人员的信息报告，例如：生成张三的人员信息报告", state["ai_model"], record_id=state.get("record_id", ""))
            return {"candidates": []}
        if ws:
            await self._send_progress(ws, state["ai_model"], f"📋 正在查询 {name} 的基本信息...", record_id=state.get("record_id", ""))
        try:
            rows_profile = await asyncio.to_thread(
                db.execute_query,
                "SELECT * FROM DIMENSYSTEM.PERSONNEL_PROFILE WHERE NAME = ? LIMIT 10",
                (name,)
            )
            rows_institution = await asyncio.to_thread(
                db.execute_query,
                "SELECT * FROM DIMENSYSTEM.PUBLIC_INSTITUTION WHERE NAME = ? LIMIT 10",
                (name,)
            )
        except Exception as e:
            if ws:
                await self._stream_response(ws, f"查询失败：{e}", state["ai_model"], record_id=state.get("record_id", ""))
            return {"candidates": []}

        candidates = []
        seen_idcards: set = set()
        for row in (rows_profile or []):
            ic = row.get("IDCARD", "")
            if ic and ic not in seen_idcards:
                seen_idcards.add(ic)
                candidates.append({
                    "name": row.get("NAME", name), "idcard": ic,
                    "unit": row.get("UNIT", ""), "position": row.get("CURRENTPOSITION", ""),
                    "source": "DIMENSYSTEM.PERSONNEL_PROFILE",
                    "label": "全市公职人员数据", "row": row,
                })
        for row in (rows_institution or []):
            ic = row.get("ID_CARD", "")
            if ic and ic not in seen_idcards:
                seen_idcards.add(ic)
                candidates.append({
                    "name": row.get("NAME", name), "idcard": ic,
                    "unit": row.get("UNIT", ""), "position": row.get("POSITION", ""),
                    "source": "DIMENSYSTEM.PUBLIC_INSTITUTION",
                    "label": "全市事业干部人员数据", "row": row,
                })

        if not candidates and ws:
            await self._stream_response(ws, f"未找到 {name} 的人员信息，请确认姓名是否正确。", state["ai_model"], record_id=state.get("record_id", ""))
            return {"candidates": []}

        # 单结果时直接填充 id_card 和 base_data，省去 disambiguate 节点
        if len(candidates) == 1:
            chosen = candidates[0]
            return {
                "candidates": candidates,
                "id_card": chosen["idcard"],
                "base_data": {chosen["label"]: [chosen["row"]]},
            }
        return {"candidates": candidates}

    async def _node_disambiguate(self, state: AgentState) -> dict:
        """节点: disambiguate — 发送消歧消息，挂起等待用户选择，resume 后解析序号。"""
        logger.info(">>> 节点: disambiguate（发送重名选择消息并等待用户选择）")
        candidates = state["candidates"]
        ws = self._ws_registry.get(state["session_id"])
        msg = self._build_candidate_message(candidates)
        logger.info(f"[人员报告] 发现重名 {len(candidates)} 人，发送选择消息")
        if ws:
            disambiguate_msg = {
                "type": "disambiguate",
                "data": {
                    "role": "assistant",
                    "aiModel": state["ai_model"],
                    "content": [{"type": "text", "content": msg}],
                    "candidates": [
                        {
                            "index": i + 1,
                            "name": c.get("name", ""),
                            "unit": c.get("unit", ""),
                            "position": c.get("position", ""),
                            "idcard_masked": self._mask_idcard(c["idcard"]),
                        }
                        for i, c in enumerate(candidates)
                    ],
                },
                "done": True,
            }
            await ws.send_json(disambiguate_msg)

        # 挂起，等待用户回复；resume 时从此处直接返回用户输入
        user_choice = interrupt({"candidates": candidates})

        # 解析序号
        m = re.search(r'\d+', str(user_choice).strip())
        if not m:
            if ws:
                await self._stream_response(ws, "请回复序号（如：1）\n\n" + msg, state["ai_model"], record_id=state.get("record_id", ""))
            user_choice = interrupt({"candidates": candidates})
            m = re.search(r'\d+', str(user_choice).strip())

        idx = (int(m.group()) - 1) if m else 0
        idx = max(0, min(idx, len(candidates) - 1))
        chosen = candidates[idx]
        logger.info(f"[人员报告] 用户选择第 {idx+1} 位：{chosen['name']} ({chosen['idcard']})")
        return {
            "id_card": chosen["idcard"],
            "base_data": {chosen["label"]: [chosen["row"]]},
        }

    async def _node_generate_report(self, state: AgentState) -> dict:
        logger.info(">>> 节点: generate_report（生成人员报告）")
        ws = self._ws_registry.get(state["session_id"])
        ai_model = state["ai_model"]
        name = state["name"]
        id_card = state["id_card"]
        base_data = state.get("base_data") or {}

        async def progress(text):
            if ws:
                await self._send_progress(ws, ai_model, text, record_id=state.get("record_id", ""))

        await progress(f"🗄️ 正在从多个数据源查询 {name} 的信息...")
        other_data = await asyncio.to_thread(
            self._query_person_data, id_card, _BASE_TABLE_NAMES
        )
        all_data = {**base_data, **other_data}

        data_sections = []
        for label, rows in all_data.items():
            cols = list(rows[0].keys())
            lines = ["| " + " | ".join(cols) + " |",
                     "|" + "|".join(["---"] * len(cols)) + "|"]
            for row in rows:
                lines.append("| " + " | ".join(str(row.get(c, '')) for c in cols) + " |")
            data_sections.append(f"### {label}\n" + "\n".join(lines))
        data_text = "\n\n".join(data_sections) if data_sections else "未查询到任何数据"

        schema_parts = []
        for table, label, _ in _PERSON_REPORT_TABLES:
            schema_txt = _PERSON_REPORT_SCHEMA.get(table.lower())
            if schema_txt:
                schema_parts.append(f"[{label}]\n{schema_txt}")
        schema_context = "\n\n".join(schema_parts) if schema_parts else ""

        await progress("✍️ 正在生成人员信息报告...")
        prompt = _PERSON_REPORT_PROMPT.replace("{question}", name).replace("{answer}", data_text)
        if schema_context:
            prompt = f"以下是各数据表的字段说明，供参考：\n\n{schema_context}\n\n---\n\n{prompt}"

        logger.info("=" * 60)
        logger.info(f"[人员报告] 开始流式生成，数据章节数: {len(all_data)}，prompt 长度: {len(prompt)} 字符")

        try:
            report = await self._stream_ai_to_ws(prompt, ws, ai_model, record_id=state.get("record_id", ""))
        except Exception as e:
            logger.error(f"[人员报告] 流式生成失败: {e}", exc_info=True)
            report = f"## {name} 人员信息（原始数据）\n\nAI生成报告失败（{type(e).__name__}），以下为数据库查询结果：\n\n{data_text}"
            if ws:
                await self._stream_response(ws, report, ai_model, record_id=state.get("record_id", ""))

        logger.info("=" * 60)
        return {"report": report}

    async def _node_select_tables(self, state: AgentState) -> dict:
        logger.info(">>> 节点: select_tables（选表）")
        ws = self._ws_registry.get(state["session_id"])
        if ws:
            await self._send_progress(ws, state["ai_model"], "🔍 正在分析问题，生成查询SQL...", record_id=state.get("record_id", ""))
        logger.info("=" * 60)
        logger.info("[节点1a: 选表] 输入 >>>")
        logger.info(f"  user: {state['content']}")

        domain = state.get("domain", "").strip()
        domain_cfg = _DOMAIN_TABLES_CFG.get(domain) if domain else None

        if domain_cfg:
            # 有明确 domain：直接使用配置里的表，再让 LLM 从中选出最相关的
            candidate_tables = domain_cfg.get("tables", [])
            domain_name = domain_cfg.get("name", domain)
            if candidate_tables:
                # 该领域各表的字段信息（仅限本领域表）
                schema_lines = []
                for tbl in candidate_tables:
                    cols = _COL_MAP.get(tbl, [])
                    comment = _TABLE_COMMENT.get(tbl.upper(), "")
                    header = f"{tbl}" + (f" -- {comment}" if comment else "")
                    if cols:
                        schema_lines.append(header + "\n  字段：" + " | ".join(cols))
                    else:
                        schema_lines.append(header)
                candidate_list = "\n\n".join(schema_lines)

                # 注入该领域的 QA 样例，帮助 LLM 选表
                qa_list = _DOMAIN_QA.get(domain, [])
                qa_section = ""
                if qa_list:
                    qa_lines = "\n".join(
                        f"  问：{qa['question']} → 表：{', '.join(qa['tables'])}"
                        for qa in qa_list
                    )
                    qa_section = f"\n\n【参考样例（问题→对应表）】\n{qa_lines}"
                select_prompt = (
                    f"你是一个数据库专家。当前专项领域：{domain_name}。\n"
                    f"该领域包含以下表及字段，根据用户问题选出最相关的表（每行一个完整表名，最多3张），不要任何解释：\n\n"
                    f"{candidate_list}"
                    f"{qa_section}\n\n"
                    f"如果完全找不到相关表，只返回：UNKNOWN"
                )
                try:
                    resp = await asyncio.to_thread(
                        self.llm.invoke,
                        [SystemMessage(content=select_prompt),
                         HumanMessage(content=state["content"])]
                    )
                    raw_tables = resp.content.strip()
                    # 过滤：只保留候选表中有的表名，防止 LLM 幻想其他表
                    candidate_set = {t.upper() for t in candidate_tables}
                    valid_lines = [
                        ln.strip() for ln in raw_tables.splitlines()
                        if ln.strip().upper() in candidate_set or ln.strip().upper() == "UNKNOWN"
                    ]
                    raw_tables = "\n".join(valid_lines) if valid_lines else "UNKNOWN"
                except Exception as e:
                    logger.error(f"[选表-领域限定] 失败: {e}")
                    raw_tables = "\n".join(candidate_tables[:3])
                logger.info(f"[节点1a: 选表-领域限定({domain})] 输出 <<< {raw_tables}")
            else:
                logger.warning(f"[选表] domain={domain} 配置中 tables 为空，降级为全库选表")
                domain_cfg = None  # 降级走全库逻辑

        if not domain_cfg:
            # 无 domain：全库选表
            try:
                resp = await asyncio.to_thread(
                    self.llm.invoke,
                    [SystemMessage(content=_SELECT_TABLE_PROMPT),
                     HumanMessage(content=state["content"])]
                )
                raw_tables = resp.content.strip()
            except Exception as e:
                logger.error(f"[选表] 失败: {e}")
                raw_tables = "UNKNOWN"
            logger.info(f"[节点1a: 选表-全库] 输出 <<< 选中表: {raw_tables}")

        # 权限检查：如果选中了人员表，验证角色权限
        user_roles = set(state.get("user_roles") or [])
        if not user_roles & _PERSON_DATA_ROLES:
            person_tables = {t.strip().upper() for t in raw_tables.splitlines() if t.strip()}
            protected = {t for t in person_tables if any(
                p in t for p in ["PERSONNEL_PROFILE", "PUBLIC_INSTITUTION"]
            )}
            if protected:
                logger.warning(f"[权限拦截] nl2sql 人员表被拒绝 | SessionID={state['session_id']} Roles={user_roles} Tables={protected}")
                if ws:
                    await self._stream_response(ws, "抱歉，您没有权限查询人员信息。", state["ai_model"], record_id=state.get("record_id", ""))
                return {"sql": "UNKNOWN"}
        else:
            logger.info(f"[权限通过] nl2sql | SessionID={state['session_id']} Roles={user_roles} Tables={raw_tables.strip()}")

        return {"sql": raw_tables}   # 暂存选表结果，generate_sql 节点读取

    async def _node_generate_sql(self, state: AgentState) -> dict:
        logger.info(">>> 节点: generate_sql（生成SQL）")
        raw_tables = state.get("sql", "")
        if not raw_tables or raw_tables.upper() == "UNKNOWN":
            return {"sql": "UNKNOWN"}
        selected = [t.strip() for t in raw_tables.splitlines() if t.strip()]
        schema_parts = []
        for tbl in selected:
            cols = _COL_MAP.get(tbl, [])
            if cols:
                schema_parts.append(f"表: {tbl}\n" + "\n".join(cols))
            else:
                schema_parts.append(f"表: {tbl}  (无字段信息)")
        schema_info = "\n\n".join(schema_parts)

        # 注入领域 QA 样例（few-shot）
        domain = state.get("domain", "")
        qa_examples = _DOMAIN_QA.get(domain, []) if domain else []
        if qa_examples:
            # 只取与选中表相关的样例，最多 3 条
            selected_set = {t.upper() for t in selected}
            relevant = [
                qa for qa in qa_examples
                if any(t.upper() in selected_set for t in qa.get("tables", []))
            ][:3]
            if relevant:
                examples_text = "\n\n".join(
                    f"问题：{qa['question']}\nSQL：{qa['sql']}"
                    for qa in relevant
                )
                schema_info += f"\n\n【参考示例（同领域真实SQL，字段均已验证）】\n{examples_text}"

        system2 = _GEN_SQL_PROMPT_TPL.format(schema_info=schema_info)
        logger.info(f"[节点1b: 生成SQL] 输入 >>> 选中表: {selected} few-shot示例: {len(relevant) if qa_examples else 0}条")
        try:
            resp = await asyncio.to_thread(
                self.llm.invoke,
                [SystemMessage(content=system2),
                 HumanMessage(content=state["content"])]
            )
            raw_sql = resp.content.strip()
        except Exception as e:
            logger.error(f"[生成SQL] 失败: {e}")
            raw_sql = "UNKNOWN"
        sql = self._extract_sql(raw_sql)

        # 检查 SQL 中是否出现了未授权的表（selected 以外的表）
        if sql and sql.upper() != "UNKNOWN":
            allowed = {t.upper() for t in selected}
            used_tables = re.findall(r'(?:FROM|JOIN)\s+([\w]+\.[\w]+)', sql, re.IGNORECASE)
            unauthorized = {t.upper() for t in used_tables if t.upper() not in allowed}
            if unauthorized:
                logger.warning(f"[生成SQL] 使用了未授权的表 {unauthorized}，强制重新生成")
                # 重新生成时必须带上 schema，否则 LLM 会凭空捏造列名
                fix_system = _GEN_SQL_PROMPT_TPL.format(schema_info=schema_info)
                fix_prompt = (
                    f"你上一次生成的SQL使用了不允许的表：{unauthorized}\n"
                    f"只能使用上方表结构中的表，禁止使用其他任何表。\n"
                    f"请严格按照上方字段列表重新生成SQL，只返回SQL不要解释。\n"
                    f"用户问题：{state['content']}"
                )
                try:
                    resp2 = await asyncio.to_thread(
                        self.llm.invoke,
                        [SystemMessage(content=fix_system),
                         HumanMessage(content=fix_prompt)]
                    )
                    sql = self._extract_sql(resp2.content.strip())
                    logger.info(f"[生成SQL] 重新生成后: {sql}")
                except Exception as e:
                    logger.error(f"[生成SQL] 重新生成失败: {e}")
                    sql = "UNKNOWN"

        logger.info(f"[节点1b: 生成SQL] 输出 <<< {sql}")
        logger.info("=" * 60)
        return {"sql": sql}

    async def _node_enrich_filter_values(self, state: AgentState) -> dict:
        """在执行SQL前，查询WHERE子句各过滤字段的DISTINCT值，让LLM校正过滤条件的实际值。"""
        logger.info(">>> 节点: enrich_filter_values（过滤值枚举）")
        sql = state.get("sql", "")
        if not sql or sql.upper() == "UNKNOWN":
            return {}

        # 明显是名称/编码/自由文本的字段后缀，跳过枚举（值无限，无参考意义）
        _SKIP_SUFFIXES = (
            '_NAME', '_CODE', '_NO', '_DESC', '_CONTENT', '_BRIEF',
            '_PERSON', '_REMARK', '_INFO', '_REASON', '_SUMMARY',
            'NAME', 'CODE', 'REMARK', 'CONTENT', 'DESC'
        )
        # 优先枚举的字段后缀（枚举类型可能性高）
        _ENUM_SUFFIXES = (
            '_TYPE', '_STATUS', '_FLAG', '_NATURE', '_LEVEL', '_MODE',
            'COUNTY', 'REGION', 'DISTRICT', 'AREA', 'UNIT', 'YEAR', 'MONTH'
        )

        table_pattern = re.compile(r'(?:FROM|JOIN)\s+(DIMENDOMAIN\.\w+)', re.IGNORECASE)
        tables_in_sql = table_pattern.findall(sql)

        filter_col_pattern = re.compile(
            r'\b(\w+)\s*(?:=|!=|<>|IN\s*\(|LIKE|NOT\s+LIKE)\s*[\'(]',
            re.IGNORECASE
        )
        raw_cols = filter_col_pattern.findall(sql)
        sql_keywords = {'AND', 'OR', 'NOT', 'NULL', 'IS', 'BETWEEN', 'EXISTS', 'SELECT', 'WHERE', 'FROM', 'JOIN'}
        filter_cols = [c.upper() for c in raw_cols if c.upper() not in sql_keywords]
        filter_cols = list(dict.fromkeys(filter_cols))

        if not tables_in_sql or not filter_cols:
            logger.info("[enrich_filter_values] 无可枚举的过滤字段，跳过")
            return {}

        distinct_info_parts = []
        queried = 0
        for col in filter_cols[:5]:
            # 字段名启发式：明显是名称/编码类，直接跳过
            if col.endswith(_SKIP_SUFFIXES) and not col.endswith(_ENUM_SUFFIXES):
                logger.info(f"[enrich_filter_values] 跳过非枚举字段: {col}")
                continue

            for tbl in tables_in_sql[:3]:
                # 先查 distinct 数量，判断是否值得枚举
                try:
                    count_rows = await asyncio.to_thread(
                        db.execute_query,
                        f"SELECT COUNT(DISTINCT {col}) AS cnt FROM {tbl} WHERE {col} IS NOT NULL"
                    )
                    distinct_count = int(list(count_rows[0].values())[0]) if count_rows else 0
                except Exception:
                    distinct_count = 0

                if distinct_count == 0:
                    continue  # 该列在该表不存在或无数据，尝试下一张表

                if distinct_count > 50:
                    logger.info(f"[enrich_filter_values] 跳过高基数字段: {tbl}.{col} (distinct={distinct_count})")
                    break  # 值太多，是名称/编码类字段，不枚举

                # distinct 值在合理范围内，查出具体值
                try:
                    rows = await asyncio.to_thread(
                        db.execute_query,
                        f"SELECT DISTINCT {col} FROM {tbl} WHERE {col} IS NOT NULL"
                    )
                    vals = [str(list(r.values())[0]) for r in rows if list(r.values())[0] is not None]
                    if vals:
                        distinct_info_parts.append(f"字段 {tbl}.{col} 的实际取值：{', '.join(vals)}")
                        queried += 1
                        break
                except Exception:
                    continue

        if not distinct_info_parts:
            logger.info("[enrich_filter_values] DISTINCT查询均无结果，跳过校正")
            return {}

        distinct_info = "\n".join(distinct_info_parts)
        logger.info(f"[enrich_filter_values] 查到 {queried} 个字段的实际取值，发送给LLM校正")

        correct_prompt = f"""以下是数据库中各过滤字段的实际取值，请根据这些实际值校正SQL中的过滤条件，确保字段值与数据库中完全一致。

数据库实际取值：
{distinct_info}

原SQL：
{sql}

用户问题：{state['content']}

校正要求：
- 只修改 WHERE 子句中与实际取值不匹配的字符串常量
- 严禁修改 SELECT 子句、GROUP BY、ORDER BY、LIMIT 等任何非 WHERE 部分
- 严禁改变查询的返回字段或聚合方式（如不得将 SELECT 字段改为 COUNT(*)）
- 严禁新增原 SQL 中不存在的 WHERE 条件
- 如果原SQL的过滤值已经正确，直接返回原SQL不做修改
- 只返回SQL，不要任何解释
"""
        try:
            resp = await asyncio.to_thread(
                self.llm.invoke,
                [HumanMessage(content=correct_prompt)]
            )
            corrected = self._extract_sql(resp.content.strip())
            if corrected and corrected.upper() != "UNKNOWN":
                logger.info(f"[enrich_filter_values] 校正后SQL: {corrected}")
                return {"sql": corrected}
        except Exception as e:
            logger.error(f"[enrich_filter_values] LLM校正失败: {e}")

        return {}

    async def _node_execute_sql(self, state: AgentState) -> dict:
        logger.info(">>> 节点: execute_sql（执行SQL）")
        ws = self._ws_registry.get(state["session_id"])
        sql = state.get("sql", "")
        ai_model = state["ai_model"]
        if not sql or sql.upper() == "UNKNOWN":
            if ws:
                await self._stream_response(ws, "抱歉，无法识别您的查询意图，请描述得更具体一些。", ai_model, record_id=state.get("record_id", ""))
            return {"query_results": []}
        if ws:
            await self._send_progress(ws, ai_model, "📋 已生成SQL，正在查询数据库...", record_id=state.get("record_id", ""))

        results = []
        sql_error = None

        # 从 SQL 中提取表名，构建 schema 供 _fix_sql 使用
        used_tables = re.findall(r'(?:FROM|JOIN)\s+([\w]+\.[\w]+)', sql, re.IGNORECASE)
        schema_parts = []
        for tbl in dict.fromkeys(t.upper() for t in used_tables):  # 去重保序
            cols = _COL_MAP.get(tbl, [])
            if cols:
                schema_parts.append(f"表: {tbl}\n" + "\n".join(cols))
        fix_schema = "\n\n".join(schema_parts)

        for attempt in range(2):
            logger.info("=" * 60)
            logger.info(f"[节点2: 数据库查询] 执行{'（重试）' if attempt > 0 else ''} >>> SQL: {sql}")
            try:
                results = await asyncio.to_thread(db.execute_query, sql)
                logger.info(f"[节点2: 数据库查询] 结果 <<< 返回 {len(results)} 条记录")
                logger.info("=" * 60)
                sql_error = None
                if results:
                    break
                if attempt == 0:
                    if ws:
                        await self._send_progress(ws, ai_model, "⚠️ 查询无结果，正在优化查询条件...", record_id=state.get("record_id", ""))
                    sql = await asyncio.to_thread(
                        self._fix_sql, state["content"], sql,
                        "查询返回0条记录，可能是JOIN条件不匹配，请简化为单表查询",
                        fix_schema
                    )
            except Exception as e:
                sql_error = str(e)
                logger.error(f"[节点2: 数据库查询] 失败: {e}")
                logger.info("=" * 60)
                if attempt == 0:
                    if ws:
                        await self._send_progress(ws, ai_model, "⚠️ SQL执行出错，正在修正...", record_id=state.get("record_id", ""))
                    sql = await asyncio.to_thread(
                        self._fix_sql, state["content"], sql, f"执行报错：{e}", fix_schema
                    )

        if (sql_error and not results) or not results:
            if sql_error:
                # 无效列名 / 无法解析成员 → 数据库缺少对应字段或表
                if "无效的列名" in sql_error or "无法解析的成员" in sql_error or "column" in sql_error.lower():
                    msg = "抱歉，数据库中暂时没有支撑该查询的数据字段，无法回答此问题。"
                elif "不存在" in sql_error or "不存在的表" in sql_error or "table" in sql_error.lower():
                    msg = "抱歉，数据库中暂时没有支撑该查询的数据表，无法回答此问题。"
                else:
                    msg = "抱歉，数据库中暂时没有支撑该查询的数据，无法回答此问题。"
            else:
                msg = "查询无结果，数据库中暂无符合条件的数据。"
            if ws:
                await self._stream_response(ws, msg, ai_model, record_id=state.get("record_id", ""))
        return {"query_results": results, "sql": sql}

    async def _node_answer_with_data(self, state: AgentState) -> dict:
        logger.info(">>> 节点: answer_with_data（数据分析回答）")
        ws = self._ws_registry.get(state["session_id"])
        ai_model = state["ai_model"]
        results = state.get("query_results") or []
        if not results:
            return {"report": ""}
        if ws:
            await self._send_progress(ws, ai_model, f"✅ 查询到 {len(results)} 条数据，正在分析结果...", record_id=state.get("record_id", ""))
        sample = results[:50]
        data_text = self._format_results(sample)
        total = len(results)
        truncated_note = f"（数据共 {total} 条，以下展示前 {len(sample)} 条）" if total > 50 else f"（共 {total} 条）"

        # 从当前 SQL 中提取用到的表，拼出字段说明（含枚举值），帮助 LLM 翻译编码
        sql = state.get("sql", "")
        schema_hint = ""
        if sql:
            used_tables = list(dict.fromkeys(
                t.upper() for t in re.findall(r'(?:FROM|JOIN)\s+([\w]+\.[\w]+)', sql, re.IGNORECASE)
            ))
            schema_lines = []
            for tbl in used_tables:
                cols = _COL_MAP.get(tbl, [])
                comment = _TABLE_COMMENT.get(tbl, "")
                if cols:
                    header = f"表 {tbl}" + (f"（{comment}）" if comment else "")
                    schema_lines.append(header + "：\n" + "\n".join(f"  {c}" for c in cols))
            if schema_lines:
                schema_hint = "\n\n参考表结构（含字段枚举值说明，请用中文描述代替编码数字）：\n" + "\n\n".join(schema_lines)

        prompt = f"""用户问题：{state['content']}

已从数据库查询到以下数据{truncated_note}：

{data_text}{schema_hint}

请根据以上数据，用简洁清晰的中文回答用户的问题。如果数据中有明确的列表信息，可以用表格或列表展示关键字段。如果字段值是枚举编码（如 1、2、3），必须根据上方表结构说明翻译为对应中文含义，不得直接展示编码。"""
        logger.info("=" * 60)
        logger.info(f"[节点3: 数据分析] 输入 >>> 数据行数: {total}，传给AI: {len(sample)} 条")
        answer = await self._stream_ai_to_ws(
            prompt, ws, ai_model, temperature=0.3, record_id=state.get("record_id", ""),
            history=state.get("history") or []
        )
        logger.info(f"[节点3: 数据分析] 输出 <<< 长度: {len(answer)} 字符")
        logger.info("=" * 60)
        return {"report": answer}

    async def _node_normal_chat(self, state: AgentState) -> dict:
        logger.info(">>> 节点: normal_chat（普通对话）")
        ws = self._ws_registry.get(state["session_id"])
        logger.info("=" * 60)
        logger.info(f"[节点1: 普通对话] 输入 >>> {state['content']}")
        answer = await self._stream_ai_to_ws(
            state["content"], ws, state["ai_model"],
            temperature=0.7, record_id=state.get("record_id", ""),
            history=state.get("history") or []
        )
        logger.info(f"[节点1: 普通对话] 输出 <<< 长度: {len(answer)} 字符")
        logger.info("=" * 60)
        return {"report": answer}

    async def _node_knowledge(self, state: AgentState) -> dict:
        logger.info(">>> 节点: knowledge（知识库）")
        ws = self._ws_registry.get(state["session_id"])
        content = state.get("content", "")
        ai_model = state.get("ai_model", "")

        try:
            from app.knowledge_service import get_knowledge_service
            ks = get_knowledge_service()
            context = ks.build_context(content, top_k=5)
        except Exception as e:
            logger.warning(f"知识库检索失败，降级为普通回答: {e}")
            context = ""

        if context:
            prompt = (
                f"请根据以下知识库内容回答用户问题。\n\n"
                f"知识库内容：\n{context}\n\n"
                f"用户问题：{content}\n\n"
                f"请基于知识库内容给出准确回答，如果知识库中没有相关信息，请如实说明。"
            )
        else:
            prompt = content

        response = await self._stream_ai_to_ws(prompt, ws, ai_model, record_id=state.get("record_id", ""))
        return {"report": response}

    # ── 通用 domain 报告节点 ──────────────────────────────────

    async def _node_classify_domain_subtype(self, state: AgentState) -> dict:
        """识别报告子类型和目标对象。
        - med_fund_mgmt：识别 area / hospital / department，提取目标名称
        - 其他领域：子类型固定为 default，目标为空
        """
        logger.info(">>> 节点: classify_domain_subtype（报告子类型识别）")
        domain = state.get("domain", "")
        if domain != "med_fund_mgmt":
            return {"domain_subtype": "default", "domain_target": ""}

        prompt = (
            "从用户输入中提取医保报告类型和目标对象。\n\n"
            "报告类型：\n"
            "- area：区域/区县整体报告，不针对特定医院或科室\n"
            "- hospital：特定医院的报告\n"
            "- department：特定科室的报告\n\n"
            "只返回 JSON，格式：{\"type\": \"area|hospital|department\", \"target\": \"目标名称或空字符串\"}\n\n"
            f"用户输入：{state['content']}"
        )
        try:
            resp = await asyncio.to_thread(self.fast_llm.invoke, [HumanMessage(content=prompt)])
            raw = resp.content.strip()
            m = re.search(r'\{.*\}', raw, re.DOTALL)
            parsed = json.loads(m.group()) if m else {}
            subtype = parsed.get("type", "area")
            target = parsed.get("target", "")
            if subtype not in ("area", "hospital", "department"):
                subtype = "area"
        except Exception as e:
            logger.error(f"[med_fund_mgmt] 子类型识别失败: {e}")
            subtype, target = "area", ""
        logger.info(f"[med_fund_mgmt] 子类型: {subtype}，目标: {target or '全市'}")
        return {"domain_subtype": subtype, "domain_target": target}

    async def _node_confirm_domain_target(self, state: AgentState) -> dict:
        """确认目标对象。
        - med_fund_mgmt + hospital 子类型：处理医院别称，必要时让用户选择
        - 其他情况：直通
        """
        logger.info(">>> 节点: confirm_domain_target（目标对象确认）")
        domain = state.get("domain", "")
        subtype = state.get("domain_subtype", "default")
        if domain != "med_fund_mgmt" or subtype != "hospital":
            return {}

        target = (state.get("domain_target") or "").strip()
        if not target:
            return {}

        matched_list = _find_hospitals_by_alias(target)
        if not matched_list:
            return {}  # 已是全称或无别称匹配

        if len(matched_list) == 1:
            full_name = matched_list[0]["full_name"]
            logger.info(f"[医院确认] 别称 '{target}' 唯一匹配 → {full_name}")
            return {"domain_target": full_name}

        ws = self._ws_registry.get(state["session_id"])
        lines = [f"找到 {len(matched_list)} 家医院，请回复序号选择："]
        for i, h in enumerate(matched_list, 1):
            lines.append(f"{i}. {h['full_name']}")
        if ws:
            await self._stream_response(ws, "\n".join(lines), state["ai_model"], record_id=state.get("record_id", ""))

        user_reply = interrupt({"hospital_candidates": [h["full_name"] for h in matched_list]})
        reply_str = str(user_reply).strip()
        m = re.search(r'\d+', reply_str)
        if m:
            idx = max(0, min(int(m.group()) - 1, len(matched_list) - 1))
            full_name = matched_list[idx]["full_name"]
        else:
            full_name = reply_str
        logger.info(f"[医院确认] 确认目标: {full_name}")
        return {"domain_target": full_name}

    async def _node_build_domain_schema(self, state: AgentState) -> dict:
        logger.info(">>> 节点: build_domain_schema（构建领域表结构）")
        domain = state.get("domain", "")
        subtype = state.get("domain_subtype") or "default"
        cfg = _DOMAIN_REPORT_CONFIGS.get(domain, {}).get(subtype, {})
        tables = [item["table"] for item in cfg.get("tables", [])]

        if not tables:
            logger.warning(f"[build_domain_schema] {domain}/{subtype} 无表配置，schema 为空")
            return {"domain_report_schema": ""}

        schema_parts = []
        for tbl in tables:
            cols = _COL_MAP.get(tbl, [])
            comment = _TABLE_COMMENT.get(tbl.upper(), "")
            header = f"表: {tbl}" + (f" -- {comment}" if comment else "")
            if cols:
                schema_parts.append(header + "\n" + "\n".join(cols))
            else:
                schema_parts.append(header + "  (无字段信息)")

        schema = "\n\n".join(schema_parts)
        logger.info(f"[build_domain_schema] {domain} schema 构建完成，共 {len(tables)} 张表")
        return {"domain_report_schema": schema}

    async def _node_generate_domain_sql(self, state: AgentState) -> dict:
        logger.info(">>> 节点: generate_domain_sql（生成聚合SQL）")
        ws = self._ws_registry.get(state["session_id"])
        if ws:
            await self._send_progress(ws, state["ai_model"], "🔍 正在分析数据，生成聚合查询...", record_id=state.get("record_id", ""))
        domain = state["domain"]
        subtype = state.get("domain_subtype") or "default"
        target = state.get("domain_target") or ""
        schema = state.get("domain_report_schema") or ""
        pkey = domain if subtype == "default" else f"{domain}_{subtype}"
        prompt_preview = _DOMAIN_REPORT_PROMPTS.get(pkey, "")[:300]
        filter_hint = (
            f"目标筛选：{target}（在 WHERE 子句中过滤相关列）"
            if target else "目标：全市整体，不需要筛选特定机构"
        )
        sql_prompt = (
            "你是达梦数据库SQL专家，使用达梦数据库SQL语法。\n"
            "根据以下数据表结构和报告模板，生成聚合查询 SQL。\n\n"
            f"【数据表结构】\n{schema}\n\n"
            f"【报告模板结构参考（节选）】\n{prompt_preview}\n\n"
            f"【{filter_hint}】\n\n"
            "要求：\n"
            "1. 生成 3-5 条 SQL，覆盖报告所需的核心指标\n"
            "2. 每条 SQL 必须是聚合查询（GROUP BY / COUNT / SUM / AVG），禁止返回明细行\n"
            "3. 如需筛选目标，用 WHERE 子句，字符串用单引号，支持 LIKE 模糊匹配\n"
            "4. 列名和表名使用数据表结构中实际存在的名称，表名必须带 schema 前缀（如 DIMENDOMAIN.XXX）\n"
            "5. 只返回 SQL 语句，每条以分号结尾，不要任何解释\n"
        )
        try:
            resp = await asyncio.wait_for(
                asyncio.to_thread(self.llm.invoke, [HumanMessage(content=sql_prompt)]),
                timeout=90
            )
            raw = resp.content.strip()
            raw = re.sub(r'```(?:sql)?', '', raw, flags=re.IGNORECASE).replace('```', '')
            sqls = [s.strip() for s in raw.split(';') if s.strip()]
        except asyncio.TimeoutError:
            logger.error(f"[{domain}/{subtype}] SQL 生成超时（90s）")
            sqls = []
        except Exception as e:
            logger.error(f"[{domain}/{subtype}] SQL 生成失败: {e}")
            sqls = []
        logger.info(f"[{domain}/{subtype}] 生成 {len(sqls)} 条聚合 SQL")
        return {"domain_report_sqls": sqls}

    async def _node_execute_domain_sql(self, state: AgentState) -> dict:
        logger.info(">>> 节点: execute_domain_sql（执行聚合SQL）")
        ws = self._ws_registry.get(state["session_id"])
        if ws:
            await self._send_progress(ws, state["ai_model"], "📊 正在聚合数据...", record_id=state.get("record_id", ""))
        sqls = state.get("domain_report_sqls") or []
        domain = state["domain"]
        subtype = state.get("domain_subtype") or "default"
        answer, extra = await asyncio.to_thread(self._exec_domain_sqls, sqls, domain, subtype)
        logger.info(f"[{domain}/{subtype}] answer 长度: {len(answer)}，extra 长度: {len(extra)}")
        return {"domain_report_answer": answer, "domain_report_extra": extra}

    async def _node_generate_domain_report(self, state: AgentState) -> dict:
        logger.info(">>> 节点: generate_domain_report（生成分析报告）")
        ws = self._ws_registry.get(state["session_id"])
        ai_model = state["ai_model"]
        if ws:
            await self._send_progress(ws, ai_model, "✍️ 正在生成分析报告...", record_id=state.get("record_id", ""))
        domain = state["domain"]
        subtype = state.get("domain_subtype") or "default"
        target = state.get("domain_target") or ""
        answer = state.get("domain_report_answer") or "无数据"
        extra = state.get("domain_report_extra") or ""
        search_condition = f"目标：{target}" if target else "全市整体"
        pkey = domain if subtype == "default" else f"{domain}_{subtype}"
        prompt = (
            _DOMAIN_REPORT_PROMPTS.get(pkey, "")
            .replace("{query}", state["content"])
            .replace("{searchCondition}", search_condition)
            .replace("{answer}", answer)
            .replace("{extraInfo}", extra)
        )
        logger.info(f"[{domain}/{subtype}] 开始流式生成报告，prompt 长度: {len(prompt)} 字符")
        try:
            report = await self._stream_ai_to_ws(
                prompt, ws, ai_model, temperature=0.3, record_id=state.get("record_id", ""),
                history=state.get("history") or []
            )
        except Exception as e:
            logger.error(f"[{domain}/{subtype}] 报告生成失败: {e}", exc_info=True)
            report = f"报告生成失败（{type(e).__name__}），原始数据：\n\n{answer}"
            if ws:
                await self._stream_response(ws, report, ai_model, record_id=state.get("record_id", ""))
        return {"report": report}

    def _exec_domain_sqls(self, sqls: list, domain: str = "", subtype: str = "default") -> tuple:
        """执行聚合 SQL 列表（达梦数据库），返回 (answer, extra)。"""
        sections = []
        for i, sql in enumerate(sqls):
            try:
                rows = db.execute_query(sql)
                if not rows:
                    sections.append(f"[查询{i+1}] 无数据\nSQL: {sql}")
                    continue
                cols = list(rows[0].keys())
                lines = [
                    f"[查询{i+1}] 共{len(rows)}条",
                    f"SQL: {sql}",
                    "| " + " | ".join(cols) + " |",
                    "|" + "|".join(["---"] * len(cols)) + "|",
                ]
                for row in rows[:50]:
                    lines.append("| " + " | ".join(str(row.get(c, "")) for c in cols) + " |")
                sections.append("\n".join(lines))
                logger.info(f"[domain_sql] 查询{i+1} 返回 {len(rows)} 条")
            except Exception as e:
                sections.append(f"[查询{i+1}] 执行失败: {e}\nSQL: {sql}")
                logger.warning(f"[domain_sql] 查询{i+1} 失败: {e}")

        answer = "\n\n".join(sections[:3]) if sections else "无数据"
        extra = "\n\n".join(sections[3:]) if len(sections) > 3 else ""
        return answer, extra

    def _fix_sql(self, question: str, bad_sql: str, error_hint: str, schema_info: str = "") -> str:
        schema_section = f"\n可用表结构（只能使用以下字段，禁止使用不存在的列名）：\n{schema_info}\n" if schema_info else ""
        prompt = f"""以下SQL执行出现问题，请修正后返回正确的SQL（只返回SQL，不要解释）。

用户问题：{question}
{schema_section}
原SQL：
{bad_sql}

问题：{error_hint}

修正要求：
- 只能使用上方表结构中实际存在的列名，禁止使用表结构中没有的列
- 如果原SQL使用了不存在的列，必须换用表中实际有的列来回答问题
- 优先改为单表查询，去掉不必要的 JOIN
- 保持 LIMIT 100
- 表名带 schema 前缀
"""
        logger.info(f"[SQL修正] 输入: {error_hint}")
        resp = self.llm.invoke([HumanMessage(content=prompt)])
        fixed = self._extract_sql(resp.content.strip())
        logger.info(f"[SQL修正] 输出: {fixed}")
        return fixed

    # ── Graph 构建 ────────────────────────────────────────────

    def _build_graph(self):
        g = StateGraph(AgentState)

        g.add_node("classify_intent",          self._node_classify_intent)
        g.add_node("extract_name",             self._node_extract_name)
        g.add_node("lookup_person",            self._node_lookup_person)
        g.add_node("disambiguate",             self._node_disambiguate)
        g.add_node("generate_report",          self._node_generate_report)
        g.add_node("select_tables",            self._node_select_tables)
        g.add_node("generate_sql",             self._node_generate_sql)
        g.add_node("enrich_filter_values",     self._node_enrich_filter_values)
        g.add_node("execute_sql",              self._node_execute_sql)
        g.add_node("answer_with_data",         self._node_answer_with_data)
        g.add_node("normal_chat",              self._node_normal_chat)
        g.add_node("knowledge",                self._node_knowledge)
        # 通用 domain 报告节点（含医保）
        g.add_node("classify_domain_subtype",  self._node_classify_domain_subtype)
        g.add_node("confirm_domain_target",    self._node_confirm_domain_target)
        g.add_node("build_domain_schema",      self._node_build_domain_schema)
        g.add_node("generate_domain_sql",      self._node_generate_domain_sql)
        g.add_node("execute_domain_sql",       self._node_execute_domain_sql)
        g.add_node("generate_domain_report",   self._node_generate_domain_report)

        g.set_entry_point("classify_intent")

        def route_intent(state: AgentState) -> str:
            intent = state.get("intent", "normal")
            return intent

        g.add_conditional_edges("classify_intent", route_intent, {
            "nl2sql":        "select_tables",
            "person_report": "extract_name",
            "domain_report": "classify_domain_subtype",
            "knowledge":     "knowledge",
            "normal":        "normal_chat",
        })

        # 人员报告路由
        g.add_edge("extract_name", "lookup_person")

        def route_person_lookup(state: AgentState) -> str:
            candidates = state.get("candidates") or []
            if not candidates:
                return "not_found"
            if len(candidates) == 1:
                return "single"
            return "multiple"

        g.add_conditional_edges("lookup_person", route_person_lookup, {
            "single":    "generate_report",
            "multiple":  "disambiguate",
            "not_found": END,
        })
        g.add_edge("disambiguate", "generate_report")
        g.add_edge("generate_report", END)

        # NL2SQL 路由
        g.add_edge("select_tables",          "generate_sql")
        g.add_edge("generate_sql",           "enrich_filter_values")
        g.add_edge("enrich_filter_values",   "execute_sql")
        g.add_edge("execute_sql",            "answer_with_data")
        g.add_edge("answer_with_data", END)

        # 通用 domain 报告路由（含医保）
        g.add_edge("classify_domain_subtype",  "confirm_domain_target")
        g.add_edge("confirm_domain_target",    "build_domain_schema")
        g.add_edge("build_domain_schema",      "generate_domain_sql")
        g.add_edge("generate_domain_sql",      "execute_domain_sql")
        g.add_edge("execute_domain_sql",       "generate_domain_report")
        g.add_edge("generate_domain_report",   END)

        g.add_edge("normal_chat", END)
        g.add_edge("knowledge",   END)

        compiled = g.compile(checkpointer=self.checkpointer)
        logger.info("[AgentService] StateGraph compiled successfully")
        return compiled

    # ── 主入口 ────────────────────────────────────────────────

    def cleanup_session(self, session_id: str):
        """清理会话状态（中止时调用）"""
        self._ws_registry.pop(session_id, None)
        # 标记为已中止，下次发消息时强制新建图而不是 resume 挂起状态
        self._aborted_sessions.add(session_id)

    async def stream_chat(self, websocket, content: str, ai_model: str, session_id: str, domain: str = "", record_id: str = "", user_roles: list = None):
        from langgraph.errors import GraphInterrupt
        logger.info("=" * 60)
        logger.info(f"[用户输入] {content}")

        self._ws_registry[session_id] = websocket
        # record_id 存入独立字典，不进 graph state 避免序列化问题
        self._record_id_registry: dict
        if not hasattr(self, '_record_id_registry'):
            self._record_id_registry = {}
        self._record_id_registry[session_id] = record_id
        user_roles = user_roles or []
        graph_config = {"configurable": {"thread_id": session_id}}

        try:
            # 从数据库读取最近 3 轮（6条）对话历史，按时间正序排列
            history: list = []
            try:
                recent = db.get_chat_records(session_id, limit=6, offset=0)
                for r in reversed(recent):
                    role = r.get("ROLE", "")
                    text = r.get("CONTENT", "") or ""
                    if role in ("user", "assistant") and text.strip():
                        history.append({"role": role, "content": text[:500]})
            except Exception as e:
                logger.warning(f"[历史上下文] 读取失败: {e}")

            # 如果会话被中止过，忽略 graph 中残留的挂起状态，强制新建
            force_new = session_id in self._aborted_sessions
            self._aborted_sessions.discard(session_id)

            state_snapshot = self.graph.get_state(graph_config)
            if state_snapshot.next and not force_new:
                logger.info("[意图识别] 恢复 interrupt（重名确认）")
                result = await self.graph.ainvoke(Command(resume=content), graph_config)
            else:
                result = await self.graph.ainvoke(
                    {
                        "session_id": session_id,
                        "ai_model": ai_model,
                        "content": content,
                        "record_id": record_id,
                        "user_roles": user_roles,
                        "history": history,
                        "intent": "", "name": "", "id_card": "",
                        "candidates": [], "base_data": {},
                        "sql": "", "query_results": [], "report": "",
                        "domain": domain,
                        "domain_subtype": "", "domain_target": "",
                        "domain_report_schema": "", "domain_report_sqls": [],
                        "domain_report_answer": "", "domain_report_extra": "",
                    },
                    graph_config
                )
        except GraphInterrupt:
            # 图已挂起等待用户输入，保留 ws_registry 以便下次 resume 时推送
            logger.info("[AgentService] 图执行已挂起（interrupt），等待用户回复")
            raise
        except Exception:
            self._ws_registry.pop(session_id, None)
            raise

        # 正常完成，清理 ws 并保存记录
        self._ws_registry.pop(session_id, None)
        response = (result or {}).get("report") or ""
        assistant_record = AiChatRecord(
            chat_id=generate_chat_id(session_id),
            session_id=session_id,
            ai_model=ai_model,
            role="assistant",
            content=response,
            create_time=get_current_timestamp(),
        )
        await self.save_chat_record(assistant_record)
