"""
泛雅学习通数据同步适配器 — 预留接口

设计意图：
  超星泛雅集团是本次大赛出题方，泛雅学习通是面向全国高校的教学平台。
  本适配器预留了从泛雅开放平台拉取真实政策文档、考研数据、心理科普
  文章的接口，体现产品的长远规模化规划——不只是为单一学校定制，
  而是可以接入泛雅生态、服务全国高校。

当前状态：
  接口已定义，参数已明确，等待泛雅开放平台 API 文档和认证凭据后即可实现。
  所有方法返回 status="reserved"，并说明需要的参数和对接步骤。

对接计划（按优先级）：
  1. 政策文档同步（policy）——拉取各校就业/资助政策 PDF/Word
  2. 考研数据同步（exam）——拉取各校历年分数线、报录比 Excel
  3. 心理科普同步（psychology）——拉取心理自助指南等文章
"""

import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class FanyaAdapter:
    """泛雅学习通平台数据同步适配器。

    对接泛雅开放平台 API，实现：
    - 政策文档自动拉取与索引
    - 考研结构化数据同步到 SQLite
    - 心理科普文档导入
    - 多校知识库隔离管理
    """

    # 预定义的同步类型
    SYNC_TYPES = {
        "policy": {
            "name": "政策文档同步",
            "description": "拉取就业政策、资助政策、三方协议等 PDF/Word 文档",
            "expected_api": "/api/v1/school/policy-documents",
            "target_collection": "policy",
        },
        "exam": {
            "name": "考研数据同步",
            "description": "拉取各校历年分数线、报录比等结构化 Excel 数据",
            "expected_api": "/api/v1/school/exam-data",
            "target_collection": "exam",
        },
        "psychology": {
            "name": "心理科普同步",
            "description": "拉取大学生心理自助指南、科普文章等",
            "expected_api": "/api/v1/school/psychology-resources",
            "target_collection": "psychology",
        },
    }

    def __init__(self, base_url: str = None, api_key: str = None):
        self.base_url = base_url
        self.api_key = api_key
        self._connected = False

    # ===== 同步入口（各类型统一接口） =====

    def sync_all(self, api_url: str, api_key: str) -> dict:
        """全量同步：拉取所有类型数据并入库。

        Args:
            api_url: 泛雅开放平台 base URL
            api_key: API 认证密钥

        Returns:
            dict: {status, synced: {policy, exam, psychology}, errors: [...]}
        """
        raise NotImplementedError(
            "泛雅适配器尚未实现。预期对接流程：\n"
            "1. 配置泛雅开放平台 API URL 和 Key\n"
            "2. 调用 /api/v1/school/* 系列端点拉取数据\n"
            "3. 文档自动导入向量库，Excel 自动入 SQLite\n"
            "4. 返回同步摘要"
        )

    def sync_policy(self, api_url: str, api_key: str) -> dict:
        """同步政策文档。

        Expected API: GET {api_url}/api/v1/school/policy-documents
        Response: [{id, title, file_url, file_type, category, updated_at}]
        """
        raise NotImplementedError(
            "政策文档同步待实现。\n"
            "需接入泛雅开放平台 /api/v1/school/policy-documents 端点"
        )

    def sync_exam_data(self, api_url: str, api_key: str) -> dict:
        """同步考研数据。

        Expected API: GET {api_url}/api/v1/school/exam-data
        Response: {schools: [{name, majors: [{name, years: [{year, score_line, ...}]}]}]}
        """
        raise NotImplementedError(
            "考研数据同步待实现。\n"
            "需接入泛雅开放平台 /api/v1/school/exam-data 端点"
        )

    def sync_psychology(self, api_url: str, api_key: str) -> dict:
        """同步心理科普文档。

        Expected API: GET {api_url}/api/v1/school/psychology-resources
        Response: [{id, title, content, category, updated_at}]
        """
        raise NotImplementedError(
            "心理科普同步待实现。\n"
            "需接入泛雅开放平台 /api/v1/school/psychology-resources 端点"
        )

    # ===== 多校管理 =====

    def list_schools(self, api_url: str, api_key: str) -> list:
        """获取泛雅平台上已接入的高校列表。

        Expected API: GET {api_url}/api/v1/schools
        Response: [{id, name, short_name, sync_status}]
        """
        raise NotImplementedError(
            "多校列表待实现。\n"
            "需接入泛雅开放平台 /api/v1/schools 端点"
        )

    def switch_school(self, school_id: str) -> dict:
        """切换当前服务的学校——切换知识库路径、校名、热线等。

        与 config/schools.yaml 联动：
        1. 从泛雅拉取该校配置
        2. 更新 schools.yaml 的 active_school
        3. 重新加载知识库
        """
        raise NotImplementedError(
            "多校切换待实现。\n"
            "需泛雅平台提供学校配置 API，配合 schools.yaml 切换"
        )

    # ===== 状态查询 =====

    def get_status(self) -> dict:
        """返回适配器当前状态——所有接口均返回 reserved，说明参数需求。"""
        return {
            "status": "reserved",
            "adapter": "FanyaAdapter v1.0",
            "description": "泛雅学习通数据同步适配器——接口已预留，等待泛雅开放平台 API 对接",
            "sync_types": {
                key: {
                    "name": val["name"],
                    "description": val["description"],
                    "expected_api": val["expected_api"],
                    "implemented": False,
                }
                for key, val in self.SYNC_TYPES.items()
            },
            "required_params": {
                "api_url": "泛雅开放平台 Base URL（如 https://open.fanya.chaoxing.com）",
                "api_key": "API 认证密钥（从泛雅开放平台获取）",
                "sync_type": "同步类型：policy / exam / psychology",
            },
            "integration_steps": [
                "1. 在泛雅开放平台注册应用，获取 api_key",
                "2. 配置 schools.yaml 中对应学校的 fanya_school_id",
                "3. 调用 POST /api/fanya/sync 触发同步",
                "4. 文档自动入库，可通过 /api/stats 查看同步结果",
            ],
            "design_rationale": (
                "本适配器体现了产品的长远规划——不只是为单一学校定制，"
                "而是可以接入泛雅学习通生态，服务全国高校。"
                "对接完成后，任何接入泛雅平台的高校都可以一键部署本助手。"
            ),
        }

    def health_check(self) -> dict:
        """检查与泛雅平台的连通性（待实现）。"""
        if not self.base_url or not self.api_key:
            return {
                "status": "not_configured",
                "message": "泛雅 API 未配置。请在 .env 中设置 FANYA_API_URL 和 FANYA_API_KEY",
            }
        return {
            "status": "reserved",
            "message": "连通性检查待实现——需泛雅开放平台端点就绪",
        }


# ===== 便捷函数 =====

_fanya_adapter: Optional[FanyaAdapter] = None


def get_fanya_adapter() -> FanyaAdapter:
    """获取全局 FanyaAdapter 单例。"""
    global _fanya_adapter
    if _fanya_adapter is None:
        _fanya_adapter = FanyaAdapter()
    return _fanya_adapter


if __name__ == "__main__":
    adapter = FanyaAdapter()
    import json
    print(json.dumps(adapter.get_status(), ensure_ascii=False, indent=2))
