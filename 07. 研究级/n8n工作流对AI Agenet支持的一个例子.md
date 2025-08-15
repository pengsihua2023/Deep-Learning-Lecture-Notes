## n8n工作流对AI Agenet支持的一个例子
n8n是一个开源的工作流自动化平台，专为技术团队设计，结合了AI功能和业务流程自动化。它允许用户以无代码速度和代码灵活性构建AI解决方案，能够与任何应用或API集成。自2019年推出以来，n8n迅速流行，尤其在2025年，它被评为欧洲新独角兽企业，估值超过10亿美元。这得益于其在AI代理和自动化领域的创新，如多代理工作流和自托管能力，使其成为开发者、IT运营和AI爱好者的首选工具。

### n8n的详细介绍
#### 核心概念
- **工作流自动化**：n8n使用节点（nodes）构建工作流，每个节点代表一个动作，如HTTP请求、数据库查询或AI调用。用户可以通过拖拽界面或代码自定义工作流，支持分支、循环和错误处理。
- **AI集成**：n8n原生支持AI代理构建，例如使用“AI Agent Tool”节点将复杂提示拆分为专注任务，支持多代理系统。用户可以集成任意LLM（如GPT系列），并通过聊天界面（如Slack或语音）创建任务或完整工作流。
- **自托管与云选项**：完全自托管（使用Docker），包括AI模型，确保数据隐私。云版本提供便利，但自托管是其核心卖点，支持企业级部署。
- **灵活开发**：结合视觉构建和自定义代码，支持JavaScript/Python、npm/PyPI库、cURL导入和分支合并。调试工具包括单步重跑、数据回放和内联日志。
- **集成**：超过500个应用集成（如Salesforce、Zoom、Asana），加上自定义节点。社区提供1700+模板，覆盖工程、文档操作等类别。

#### 关键特性
- **企业就绪**：SSO（SAML/LDAP）、加密秘密存储、版本控制、RBAC权限、审计日志、工作流历史和协作工具（如Git集成）。
- **性能与安全性**：自托管保护数据，支持白标自动化（保持品牌）。
- **社区与资源**：活跃GitHub仓库、论坛和教程。社区分享作弊表（如德语/英语版本）和模板。2025年有黑客马拉松（如与GPT-5集成）和视频教程。
- **用例**：自动化客户数据管理、发票提醒、Salesforce更新、Asana任务创建。案例：Delivery Hero每月节省200小时，StepStone将两周工作减至2小时。

#### 最近发展（2025年）
- **独角兽地位**：n8n在2025年成为欧洲新独角兽，通过AI代理简化工作流自动化。
- **AI增强**：新节点如AI Agent Tool，支持多代理系统和GPT-5集成。黑客马拉松和教程聚焦AI代理构建。
- **社区活跃**：新作弊表、会议（如n8n社区聚会）和模板发布。用户分享自托管经验和成本优化。
- **竞争与替代**：与Zapier、String比较，n8n强调开源和自托管。2025年有视频比较其速度和Web3集成。

#### 定价
n8n的核心是免费开源（fair-code许可）。云版本有免费层（有限执行）和付费计划（从每月20美元起，支持无限工作流）。企业版自定义定价，包括高级支持。

### 代码例子
n8n工作流以JSON格式定义，可导入界面。下面是一个基于真实用例的简单例子：从RSS源获取新闻，使用AI总结并发送到Slack。

#### JSON工作流例子
```json
{
  "name": "RSS to AI Summary to Slack",
  "nodes": [
    {
      "parameters": {
        "url": "https://example.com/rss"
      },
      "name": "RSS Feed",
      "type": "n8n-nodes-base.rss",
      "typeVersion": 1,
      "position": [240, 300]
    },
    {
      "parameters": {
        "model": "gpt-4",
        "prompt": "Summarize this article: {{ $json.content }}"
      },
      "name": "AI Summary",
      "type": "n8n-nodes-base.aiAgent",
      "typeVersion": 1,
      "position": [460, 300]
    },
    {
      "parameters": {
        "channel": "#news",
        "text": "{{ $json.summary }}"
      },
      "name": "Send to Slack",
      "type": "n8n-nodes-base.slack",
      "typeVersion": 1,
      "position": [680, 300],
      "credentials": {
        "slackApi": "Your Slack Credentials"
      }
    }
  ],
  "connections": {
    "RSS Feed": {
      "main": [
        [
          {
            "node": "AI Summary",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "AI Summary": {
      "main": [
        [
          {
            "node": "Send to Slack",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```
**说明**：
1. **导入步骤**：在n8n界面，复制JSON并导入工作流。
2. **运行**：设置凭证（Slack API密钥），激活工作流。它每小时检查RSS，AI总结内容，并发送到Slack频道。
3. **自定义**：替换URL、模型和通道。使用n8n的AI Agent节点集成LLM。

更多例子可在n8n社区（https://n8n.io/workflows/）找到4637+模板。
