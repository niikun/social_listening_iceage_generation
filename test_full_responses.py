#!/usr/bin/env python3
# テスト用スクリプト：全回答表示機能のテスト

import sys
import os
import asyncio
import pandas as pd
sys.path.append(os.path.dirname(__file__))

from app import SimulationProvider, JapanDemographicsDB, PersonaGenerator

async def test_full_response_display():
    print("=== 全回答表示機能テスト ===")
    
    # データベースとジェネレーターを初期化
    demographics_db = JapanDemographicsDB()
    persona_generator = PersonaGenerator(demographics_db)
    provider = SimulationProvider()
    
    # 5人のペルソナを生成
    personas = []
    for i in range(5):
        persona = persona_generator.generate_persona(i + 1)
        persona_dict = {
            'id': persona.id,
            'age': persona.age,
            'gender': persona.gender,
            'generation': persona.generation,
            'occupation': persona.occupation,
            'education': persona.education,
            'income_level': persona.income_level,
            'family_status': persona.family_status,
            'media_preferences': persona.media_preferences
        }
        personas.append(persona_dict)
    
    # テスト質問
    question = "働き方改革についてどう思いますか？"
    
    print(f"\n【質問】{question}")
    print("="*80)
    
    # 各ペルソナから回答を取得
    responses = []
    for persona in personas:
        result = await provider.generate_response(persona, question)
        
        response = {
            'persona_id': persona['id'],
            'persona': persona,
            'question': question,
            'response': result['response'],
            'success': result.get('success', True)
        }
        responses.append(response)
    
    # 全回答をDataFrameに変換（実際のアプリと同じ形式）
    response_df = pd.DataFrame([{
        'generation': r['persona']['generation'],
        'age': r['persona']['age'],
        'gender': r['persona']['gender'],
        'occupation': r['persona']['occupation'],
        'education': r['persona']['education'],
        'income_level': r['persona']['income_level'],
        'family_status': r['persona']['family_status'],
        'response': r['response']
    } for r in responses])
    
    print(f"\n📊 回答一覧（全{len(response_df)}件）:")
    print("="*80)
    
    # 詳細表示バージョン
    print("\n🔸 詳細表示（ペルソナ属性付き）:")
    for idx, (_, row) in enumerate(response_df.iterrows(), 1):
        print(f"\n--- 回答 {idx}: {row['age']}歳 {row['gender']} ({row['occupation']}) ---")
        print(f"【基本情報】")
        print(f"  年齢: {row['age']}歳")
        print(f"  性別: {row['gender']}")
        print(f"  職業: {row['occupation']}")
        print(f"【詳細属性】")
        print(f"  学歴: {row['education']}")
        print(f"  収入: {row['income_level']}")
        print(f"  家族: {row['family_status']}")
        print(f"【💬 回答】")
        print(f"  > {row['response']}")
    
    print(f"\n" + "="*80)
    print(f"\n🔸 シンプル表示（回答のみ）:")
    for idx, (_, row) in enumerate(response_df.iterrows(), 1):
        print(f"\n{idx}. {row['age']}歳 {row['gender']}")
        print(f"💬 {row['response']}")
        print("-" * 40)
    
    print(f"\n✅ 全回答表示機能テスト完了（{len(responses)}件の回答を処理）")

if __name__ == "__main__":
    asyncio.run(test_full_response_display())