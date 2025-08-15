#!/usr/bin/env python3
# テスト用スクリプト：高度な回答判定ロジックのテスト

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from app import SimulationProvider

async def test_advanced_response_logic():
    print("=== 就職氷河期世代特化回答判定テスト ===")
    
    provider = SimulationProvider()
    
    # 様々なペルソナパターン
    personas = [
        {'age': 43, 'occupation': '正規雇用', 'gender': '男性', 'generation': '就職氷河期世代'},
        {'age': 52, 'occupation': '非正規雇用', 'gender': '女性', 'generation': '就職氷河期世代'},
        {'age': 47, 'occupation': '失業・無業', 'gender': '男性', 'generation': '就職氷河期世代'},
    ]
    
    # 就職氷河期世代に特化した質問パターン
    test_questions = [
        # 雇用関連（支援系 - positive期待）
        "非正規雇用者への就職支援を充実させるべきだと思いますか？",
        "就職氷河期世代への再就職対策についてどう思いますか？",
        
        # 雇用関連（問題系 - negative期待）
        "現在の雇用不安定について、どのような問題を感じますか？",
        "派遣労働の現状についてどう思いますか？",
        
        # 経済関連（支援系 - positive期待）
        "家計負担軽減のための税制改正に賛成しますか？",
        "年金制度改善の必要性についてどう思いますか？",
        
        # 経済関連（負担系 - negative期待）
        "消費税増税による家計への影響をどう感じますか？",
        "経済格差拡大の問題についてどう思いますか？",
        
        # 働き方改革（positive期待）
        "職場環境の改善についてどう考えますか？",
        "ワークライフバランスの推進は必要だと思いますか？",
        
        # 次世代支援（positive期待）
        "若者の就職活動支援を充実させることについてどう思いますか？",
        "子どもたちの教育環境改善についてどう考えますか？",
        
        # 変化・改革（年齢により異なる期待）
        "デジタル化の進展についてどう思いますか？", # 40代:neutral, 50代:negative
        "新しい働き方への変化についてどう感じますか？", # 40代:neutral, 50代:negative
        
        # 制度・政策（neutral期待）
        "社会保障制度についてどう考えますか？",
        "政府の経済政策についてどう思いますか？"
    ]
    
    for persona in personas:
        print(f"\n{'='*80}")
        print(f"テスト対象: {persona['age']}歳 {persona['gender']} ({persona['occupation']})")
        print(f"{'='*80}")
        
        for question in test_questions:
            # 感情判定をテスト
            sentiment = provider._determine_sentiment_for_ice_age_generation(
                question, persona['age'], persona['occupation']
            )
            
            # 実際の回答生成
            result = await provider.generate_response(persona, question)
            
            print(f"\n【質問】{question}")
            print(f"判定感情: {sentiment}")
            print(f"回答: {result['response']}")
            print("-" * 60)
    
    print("\n✅ 高度な回答判定ロジックテスト完了")

if __name__ == "__main__":
    asyncio.run(test_advanced_response_logic())