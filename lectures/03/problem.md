# 第3回 演習・宿題：学習支援エージェントを作る

## この回のゴール

- エージェントが「モデル＋ループ＋ツール」でできていることを、自分のコードで確かめる
- どのツールが、どの引数で呼ばれたかを読める
- ツールを1つ追加できる

## 演習（30分）

最初の約5分で第2回の宿題を解説します。その間に、手順1の Sync fork を押し、Codespaces を起動しておきましょう（起動に数分かかります）。

### 1. 講座 repo の更新を取り込み、lec03 ブランチを作る

講座 repo の修正を自分の fork に取り込んでから、作業ブランチを作ります。

1. GitHub で自分の fork を開き、**Sync fork → Update branch** を押します（すでに最新なら Update branch は出ないので、何もしなくてかまいません）
2. Codespaces を起動し、ターミナルで次を実行します

```bash
git switch main
git pull --no-rebase
git switch -c lec03
```

- `git switch main` で止まる場合は、前の回のブランチに変更が残っています。前の回のブランチで commit してから、もう一度実行してください
- `a branch named 'lec03' already exists` と出た場合は、`lec03` はもう作ってあります。`git switch lec03` で切り替えてください
- `git pull --no-rebase` で `CONFLICT` と出た場合は、`git merge --abort` で元に戻してから、下の方法で始めてください（元に戻すので、自分の変更は消えません）

#### main が壊れた、または前の回までの作業が終わっていない場合

Sync fork で衝突の表示（Pull request の作成や「Discard commits」）が出た場合も、**押さずに**こちらで始めます。「Discard commits」を押すと、自分のコミットが消えます。

講座 repo の開始時点のコード（`lec03-start` タグ）からブランチを作ります。

```bash
git fetch https://github.com/matsuolab/lecture-ai-engineering-2026.git tag lec03-start
git switch -c lec03 lec03-start
```

### 2. 第1回と同じ質問をエージェントに投げる

```bash
uv run python -m app.agent "AIE2026の講座の宿題にある「3レベル」とは何ですか？"
```

第1回の `app.ask` の回答と比べてみましょう。エージェントは用語集を検索してから答えます。正しい内容は `data/glossary.json` の `homework-levels` にあります。

### 3. ツール呼び出しを読む

```bash
uv run python -m app.agent --trace "LoRAとQLoRAの違いは？"
```

`[tool call]` の行で、モデルがどのツールをどんな引数で呼んだかを確認します。`[usage]` の `requests` は、1つの質問で LLM を何回呼んだかを表します。

なぜ答える前に検索するのか、なぜ用語を1つずつ検索するのかは、モデルへの指示に書いてあります。指示は2か所にあります。

- `app/agent.py` の `INSTRUCTIONS`（31行目付近）：エージェント全体への指示。「必ず search_glossary で用語集を確認する」「回答の最後に、関係する回（第N回）を示す」などが書いてあります
- `app/tools.py` の各ツールの docstring：ツールの説明文としてモデルに渡ります。`search_glossary` には「query には調べたい用語を1つだけ入れる」と書いてあります

宿題の基礎では、この2か所を書き換えて、ふるまいがどう変わるかを比べます。

### 4. make_quiz ツールを書いて登録する

「第2回の確認問題を出して」と頼まれたときに、エージェントが用語集から問題を作れるようにします。モデルが `make_quiz` を `lecture=2` で呼び、ツールが第2回の用語を1つ選んで問題と答えを返し、モデルがそれを使って出題します。

1. `app/tools.py` に関数 `make_quiz(lecture: int) -> str` を追加する
2. docstring に「いつ、何のために使うツールか」を書く（モデルはこれを読んで、使うかどうかを決める）
3. 用語集（`load_glossary()`）から、`lecture` が指定の回の用語を集め、1つ選ぶ（`random.choice` など）
4. 「○○の意味は？」という問題と、答え（その用語の `definition`）を文字列で返す（`json.dumps` など）
5. `app/agent.py` の `TODO(第3回 演習)` の場所で `agent.tool_plain(tools.make_quiz)` と登録する

`make_quiz(2)` の返り値の例：

```text
{"question": "「PEFT」の意味は？", "answer": "モデルの全パラメータではなく、…"}
```

### 5. make_quiz が呼ばれるか確かめる

```bash
uv run python -m app.agent --list-tools    # モデルに見えるツールの一覧（API キー不要）
uv run python -m app.agent --trace "第2回の内容から確認問題を1問出して"
uv run pytest -q
```

一覧に `make_quiz` と自分の docstring が出て、`[tool call] make_quiz({'lecture': 2})` のような行が出れば成功です。

`uv run pytest -q` は、API キーを使わずに、追加した `make_quiz` も含めて全部のツールをダミーの引数（`lecture=0` など）で呼び、エラーで落ちないかを確かめます。第0回は存在しないので、用語が見つからないときの処理を書いていないと失敗します。そのときは、用語が無い場合に分かりやすいメッセージを返すようにしましょう。

- 一覧に無い → `app/agent.py` で登録していない
- 一覧にはあるが呼ばれない → docstring に使う場面を具体的に書く。`INSTRUCTIONS` に「確認問題を求められたら make_quiz を使う」と足す

### 6. 提出して、Codespaces を停止する

`lectures/03/answer.md` を書き、`lec03` → 自分の fork の `main` へ PR を出します。**base repository が自分の fork になっていることを確認してください。**

```bash
git add -A
git commit -m "第3回 演習"
git push -u origin lec03
```

手順4が終わらなかった場合は、手順3までの結果で提出してかまいません。続きは同じブランチに push すれば PR に反映されます。

最後に、左下の「Codespaces」→ **Stop Current Codespace** で停止します。

## 宿題（どれか1つで完了）

| レベル | 内容 |
|---|---|
| 基礎 | `INSTRUCTIONS` やツールの docstring を書き換え、ツールの呼ばれ方や回答がどう変わるかを `--trace` で比べて記録する |
| 実装 | 学習の記録を `learning_log/` に保存する `save_learning_log` ツールを作る。保存の前に利用者の確認を挟む（Pydantic AI の `requires_approval=True`）。TestModel を使ったテストも書く |
| 発展 | 次のどちらか。(a) 同じエージェントを Google の ADK で作り直し、書き方や `adk web` での見え方を比較する。(b) 講義で紹介した Pi を Codespaces 上で動かし、公開されている Pi のシステムプロンプトと自分のエージェントの `INSTRUCTIONS` を比較する |

### 発展（b）の進め方

Pi はファイルの読み書きやコマンドの実行ができるコーディングエージェントで、ツールを実行するたびに確認は求めません。自分のPCではなく、Codespaces の中で試してください。

```bash
npm install -g --ignore-scripts @earendil-works/pi-coding-agent
pi
```

- Gemini は環境変数 `GEMINI_API_KEY` を読むので、Codespaces secret があればそのまま使えます。Pi の中で `/model` と入力し、Gemini のモデルを選びます
- npm のパッケージ名は `@earendil-works/pi-coding-agent` です（古い記事にある `@mariozechner/...` は旧名です）
- Pi のシステムプロンプトは、公開リポジトリ https://github.com/earendil-works/pi の `packages/coding-agent/src/core/system-prompt.ts` にあります。自分の `INSTRUCTIONS` と比べ、何が書かれていて何が書かれていないかを見てみましょう
- 公式ドキュメント：https://github.com/earendil-works/pi/tree/main/packages/coding-agent/docs

## 参考

- Pydantic AI: https://ai.pydantic.dev/
- ADK: https://google.github.io/adk-docs/
