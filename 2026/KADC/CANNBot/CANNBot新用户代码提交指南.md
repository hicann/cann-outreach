# CANNBot 新用户代码提交指南

在 CANNBot 赛道中，欢迎新用户提交作品！

提交作品之前, 请 确保已经完成了 [CANNBot新用户快速上手指南](./CANNBot新用户快速上手指南.md)

本指南指引你将作品通过 **Fork 仓提 PR** 的方式提交到 [cann/cann-outreach](https://gitcode.com/cann/cann-outreach)。

---

## Step 1：Fork cann-outreach 仓库

1. 浏览器打开 https://gitcode.com/cann/cann-outreach
2. 点击右上角 **【Fork】** 按钮，将仓库 Fork 到你的个人账号下
3. Fork 完成后，你的个人仓库地址为：`https://gitcode.com/<你的用户名>/cann-outreach`

---

## Step 2：配置 Git 账号信息

在 WebIDE 终端中创建 `git_env.bash`，替换 `<...>` 为你的实际信息：

```bash
# 配置 Gitcode 访问令牌
GITCODE_TOKEN="你的GitCode访问令牌"

# 下面两行从 GitCode 克隆弹窗中直接粘进来
git config --global user.name xxxxxxx
git config --global user.email XAXAXA@UUU.com

# 从 git config 中自动解析用户名和邮箱
GIT_USER=$(git config --global user.name)
GIT_EMAIL=$(git config --global user.email)

sed -i '/gitcode.com/d' ~/.git-credentials
echo "https://${GIT_USER}:${GITCODE_TOKEN}@gitcode.com" >> ~/.git-credentials
```

> - 用户名和邮箱：在 GitCode 任意仓库页面点击【克隆】按钮可查看。
> - 访问令牌：在 GitCode「个人设置」→「访问令牌」中创建并获取。

执行以下命令使配置生效：

```bash
source git_env.bash
```

---

## Step 3：克隆你 Fork 的 cann-outreach

在 WebIDE 终端中，进入持久化目录并克隆：

```bash
cd /mnt/workspace/
git clone https://gitcode.com/<你的用户名>/cann-outreach.git
cd cann-outreach
```

> `/mnt/workspace/` 是持久化存储目录，环境重启后数据不会丢失。

---

## Step 4：将作品代码放入提交目录

cann-outreach 的目录结构：

```
cann-outreach/
└── 2026/KADC/
    ├── PyPTO/        # PyPTO 赛道
    ├── AscendC/      # AscendC 赛道
    └── CANNBot/      # CANNBot 赛道 ← 作品代码放这里
```

作品路径格式为 `CANNBot/<用户名>_<作品名称>`：

```bash
cp -r <你的作品代码路径> 2026/KADC/CANNBot/<你的用户名>_<作品名称>/
# 清理编译产物
rm -rf 2026/KADC/CANNBot/<你的用户名>_<作品名称>/build_out
```

示例（用户 `zhangsan` 提交 Abs 算子）：

```bash
cp -r /path/to/abs 2026/KADC/CANNBot/zhangsan_abs/
rm -rf 2026/KADC/CANNBot/zhangsan_abs/build_out
```

---

## Step 5：提交并推送

```bash
git add 2026/KADC/CANNBot/<你的用户名>_<作品名称>/
git commit -m "feat(CANNBot): add <作品名称>"
git push origin master
```

---

## Step 6：发起 Pull Request

1. 浏览器打开 `https://gitcode.com/<你的用户名>/cann-outreach`
2. 点击 **【Pull Requests】** → **【New Pull Request】**
3. 确认合并方向：
   - `base: cann/cann-outreach` `master`
   - `head: <你的用户名>/cann-outreach` `master`
4. 填写 PR 标题和描述，示例：

   **标题**：`[CANNBot] 提交 <作品名称>`

   **描述**：
   ```
   ## 作品简介
   - 名称：Abs
   - 数据类型：float16
   - 支持 shape：[1,128]、[4,2048]、[32,4096]

   ## 验证情况
   - [x] 功能验证通过
   - [x] 单元测试通过
   ```

5. 点击 **【Create Pull Request】** 提交

> 首次向 cann 组织仓库提交 PR，可能需要签署 [CLA 贡献者许可协议](https://gitcode.com/cann/infrastructure/blob/main/docs/cla/cla使用指南.md)。

---

## 流程总结

```
Fork cann-outreach → 克隆到 CANNLab → 放入作品代码 → commit → push → 发起 PR
```

PR 合入后，你的作品即出现在 [cann/cann-outreach](https://gitcode.com/cann/cann-outreach) 仓库中。
