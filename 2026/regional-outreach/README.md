# 区域创新中心活动

## 活动介绍
为支撑区域开发者更好地学习和使用昇腾AI平台，区域创新中心将联合CANN社区定期开展主题分享、技术沙龙、校园行、训练营等多种形式的活动，助力开发者快速成长。

## 目录结构
```
regional-outreach/
├── README.md                                                 # 本文档
├── jiangsu/                                                  # 江苏区域活动
│   └── CANN-Campus-Tour_Nanjing-University-Session/          # 活动名称
│       └── {gitcode用户名}/                                   # 开发者gitcode账号
│           └── {任务名称}/                                    # 贡献内容
├── shanghai/                                                 # 上海区域活动
├── beijing/                                                  # 北京区域活动
```

## 活动类型

| 活动类型 | 目录命名示例 | 说明 |
|---------|-------------|------|
| 校园行 | `CANN-Campus-Tour_Nanjing-University-Session` | 高校巡讲，走进校园开展昇腾技术分享与实操 |
| 技术沙龙 | `CANN-Tech-Salon_2026S1` | 区域技术沙龙，聚焦特定技术主题深度交流 |
| 训练营 | `CANN-Bootcamp_2026S1` | 集中培训营，系统性学习昇腾开发全流程 |
| 主题分享 | `CANN-Workshop_Custom-Op` | 专题工作坊，围绕特定场景动手实践 |

## 提交规范

### 命名规范
- **区域目录**：使用城市/省份拼音全小写，如 `jiangsu`、`shanghai`、`beijing`
- **活动目录**：格式为 `{活动类型}_{活动主题}`，单词间用连字符，各部分用下划线分隔
  - 示例：`CANN-Campus-Tour_Nanjing-University-Session`
- **用户目录**：使用 gitcode 用户名
- **任务目录**：使用任务名称，如 `TanhCustom`

### 提交步骤
1. 在对应 **区域目录** 下创建以 **活动类型_活动主题** 命名的文件夹
2. 在活动文件夹下创建以 **gitcode用户名** 命名的文件夹
3. 在用户名文件夹下创建以 **任务名称** 命名的文件夹
4. 按照任务要求提交贡献内容
5. 提交 Pull Request 到本仓库

## 联系方式
- **官方邮箱**：community@cann-community.com
- **技术支持**：support@cann-community.com
- **Issue 反馈**：https://gitcode.com/cann/cann-competitions/issues
