#!/usr/bin/env python3
# テスト用スクリプト：感情分析表示のテスト

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import ResponseAnalyzer

def test_sentiment_analysis_display():
    print("=== 感情分析表示テスト ===")
    
    analyzer = ResponseAnalyzer()
    
    # テスト用の回答データ
    test_responses = [
        "働き方改革には賛成です。良い取り組みだと思います。",
        "経済負担が心配です。不安があります。",
        "もう少し具体的な内容を知りたいです。",
        "期待しています。希望が持てます。",
        "問題があると思います。反対です。",
        "様々な角度から検討したいです。",
        "支持したいと思います。良いアイデアです。",
        "慎重に判断する必要があります。",
        "不安な点もありますが期待もしています。",
        "バランスを考えて進めてほしいです。"
    ]
    
    # 感情分析実行
    sentiment = analyzer.analyze_sentiment(test_responses)
    
    print(f"\n📊 感情分析結果:")
    print(f"ポジティブ: {sentiment['positive']:.1f}%")
    print(f"ネガティブ: {sentiment['negative']:.1f}%")
    print(f"中立: {sentiment['neutral']:.1f}%")
    
    # 新しい表示形式のテスト
    print(f"\n📈 メトリック表示:")
    def format_delta(value, base=33.3):
        if value == base:
            return "基準値"
        delta = value - base
        return f"+{delta:.1f}%" if delta > 0 else f"{delta:.1f}%"
    
    print(f"ポジティブ: {sentiment['positive']:.1f}% (差分: {format_delta(sentiment['positive'])})")
    print(f"ネガティブ: {sentiment['negative']:.1f}% (差分: {format_delta(sentiment['negative'])})")
    print(f"中立: {sentiment['neutral']:.1f}% (差分: {format_delta(sentiment['neutral'])})")
    
    # 総合判定
    print(f"\n🎯 総合判定:")
    if sentiment['positive'] > sentiment['negative'] + 10:
        print("📈 全体的にポジティブな反応")
    elif sentiment['negative'] > sentiment['positive'] + 10:
        print("📉 全体的にネガティブな反応")
    else:
        print("⚖️ バランスの取れた反応")
    
    # 詳細な分析
    print(f"\n📋 詳細分析:")
    dominant = max(sentiment, key=sentiment.get)
    print(f"最も多い感情: {dominant} ({sentiment[dominant]:.1f}%)")
    
    total_emotional = sentiment['positive'] + sentiment['negative']
    print(f"感情的な反応: {total_emotional:.1f}%")
    print(f"中立的な反応: {sentiment['neutral']:.1f}%")
    
    print(f"\n✅ 感情分析表示テスト完了")

if __name__ == "__main__":
    test_sentiment_analysis_display()