#!/usr/bin/env python3
# テスト用スクリプト：全属性を活用した個別化回答システムのテスト

import sys
import os
import asyncio
sys.path.append(os.path.dirname(__file__))

from app import SimulationProvider, JapanDemographicsDB, PersonaGenerator

async def test_personalized_responses():
    print("=== 全属性活用個別化回答システムテスト ===")
    
    # データベースとジェネレーターを初期化
    demographics_db = JapanDemographicsDB()
    persona_generator = PersonaGenerator(demographics_db)
    provider = SimulationProvider()
    
    # 多様なペルソナを生成
    test_personas = []
    for i in range(8):
        persona = persona_generator.generate_persona(i + 1)
        test_personas.append({
            'id': persona.id,
            'age': persona.age,
            'gender': persona.gender,
            'occupation': persona.occupation,
            'education': persona.education,
            'income_level': persona.income_level,
            'family_status': persona.family_status,
            'media_preferences': persona.media_preferences
        })
    
    # テスト質問
    test_questions = [
        "働き方改革についてどう思いますか？",
        "子どもの教育費支援を充実させることについてどう考えますか？",
        "年金制度の改善についてどう思いますか？",
        "デジタル化の進展についてどう感じますか？"
    ]
    
    for question in test_questions:
        print(f"\n{'='*80}")
        print(f"質問: {question}")
        print(f"{'='*80}")
        
        for persona in test_personas:
            print(f"\n【ペルソナ {persona['id']}】")
            print(f"基本情報: {persona['age']}歳 {persona['gender']} | 職業: {persona['occupation']}")
            print(f"学歴: {persona['education']} | 年収: {persona['income_level']}")
            print(f"家族構成: {persona['family_status']}")
            
            # メディア接触パターン表示
            media_contacts = []
            if persona.get('media_preferences'):
                for media, contact in persona['media_preferences'].items():
                    if contact:
                        media_names = {
                            'instagram': 'Instagram', 'x': 'X', 'youtube': 'YouTube',
                            'tv': 'TV', 'newspaper_print': '新聞(紙)', 
                            'newspaper_digital': '新聞(デジタル)'
                        }
                        media_contacts.append(media_names.get(media, media))
            print(f"メディア接触: {', '.join(media_contacts[:3]) if media_contacts else 'なし'}")
            
            # 個別化回答生成
            result = await provider.generate_response(persona, question)
            
            if result['success']:
                print(f"回答: {result['response']}")
            else:
                print(f"エラー: {result['response']}")
            
            print("-" * 60)
    
    print("\n✅ 全属性活用個別化回答システムテスト完了")

if __name__ == "__main__":
    asyncio.run(test_personalized_responses())