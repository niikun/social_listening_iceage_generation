#!/usr/bin/env python3
# テスト用スクリプト：就職氷河期世代の回答パターンテスト

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from app import SimulationProvider

async def test_response_patterns():
    print("=== 就職氷河期世代回答パターンテスト ===")
    
    provider = SimulationProvider()
    
    # テスト用ペルソナ
    test_persona = {
        'id': 1,
        'age': 48,
        'gender': '男性',
        'generation': '就職氷河期世代',
        'occupation': '正規雇用',
        'prefecture': '東京都'
    }
    
    # テスト質問
    test_questions = [
        # ポジティブなトーンの質問
        "働き方改革を推進すべきだと思いますか？",
        "若者の就職支援を充実させることについてどう思いますか？",
        
        # ネガティブなトーンの質問  
        "現在の雇用問題についてどのような不安がありますか？",
        "経済格差の課題について、どう感じますか？",
        
        # 中立的な質問
        "今後の社会保障制度についてどう考えますか？",
        "デジタル化の進展について意見をお聞かせください？"
    ]
    
    print(f"テスト対象ペルソナ: {test_persona['age']}歳 {test_persona['gender']} ({test_persona['generation']})")
    print("="*60)
    
    for i, question in enumerate(test_questions, 1):
        print(f"\n【質問 {i}】{question}")
        
        # 回答を3回生成してバリエーションを確認
        for j in range(3):
            result = await provider.generate_response(test_persona, question)
            if result['success']:
                print(f"回答{j+1}: {result['response']}")
            else:
                print(f"回答{j+1}: エラー - {result['response']}")
        
        print("-" * 40)
    
    print("\n✅ 回答パターンテスト完了")

if __name__ == "__main__":
    asyncio.run(test_response_patterns())