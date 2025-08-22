# Social Listening Iceage Generation

## 概要
このプロジェクトは、ソーシャルリスニングを活用して、特定のテーマに関連するデータを収集・分析し、インサイトを生成するためのアプリケーションです。

## ディレクトリ構成
```
app.py
CLAUDE.md
install_log.txt
install.py
requirements.txt
test_advanced_response.py
test_cost_calculation.py
test_full_responses.py
test_persona_display.py
test_personalized_responses.py
test_personas.py
test_response_patterns.py
test_sentiment_analysis.py
test_summary_methods.py
notebook/
    data/
        data.ipynb
        sy24rv10rc.csv
        sy24rv20rc.csv
```

## 必要条件
- Python 3.10以上
- 必要なライブラリは`requirements.txt`に記載されています。

## インストール
以下の手順でプロジェクトをセットアップしてください：

1. リポジトリをクローンします。
   ```bash
   git clone https://github.com/niikun/social_listening_iceage_generation.git
   ```

2. 必要なライブラリをインストールします。
   ```bash
   pip install -r requirements.txt
   ```

## 使用方法
1. アプリケーションを実行します。
   ```bash
   python app.py
   ```

2. 必要に応じて、`notebook/data/`ディレクトリ内のJupyter Notebookを使用してデータを分析してください。

## テスト
以下のコマンドでユニットテストを実行できます：
```bash
pytest
```

## ライセンス
Apache License 2.0

## 貢献
バグ報告や機能リクエストは、GitHubのIssueを通じて行ってください。プルリクエストも歓迎します。
