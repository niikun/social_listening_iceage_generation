#!/usr/bin/env python3
# テスト用スクリプト：新しいペルソナ生成機能のテスト

import sys
import os
sys.path.append(os.path.dirname(__file__))

from app import JapanDemographicsDB, PersonaGenerator

def test_persona_generation():
    print("=== ペルソナ生成テスト ===")
    
    # データベースとジェネレーターを初期化
    demographics_db = JapanDemographicsDB()
    persona_generator = PersonaGenerator(demographics_db)
    
    # 5人のペルソナを生成
    personas = []
    for i in range(5):
        persona = persona_generator.generate_persona(i + 1)
        personas.append(persona)
    
    # 結果を表示
    for persona in personas:
        print(f"\n--- ペルソナ {persona.id} ---")
        print(f"年齢: {persona.age}歳 ({persona.generation})")
        print(f"性別: {persona.gender}")
        print(f"職業: {persona.occupation}")
        print(f"学歴: {persona.education}")
        print(f"年収: {persona.income_level}")
        print(f"家族構成: {persona.family_status}")
        print(f"居住環境: {persona.urban_rural}")
        print("メディア接触:")
        if persona.media_preferences:
            for media, contact in persona.media_preferences.items():
                status = "✓" if contact else "✗"
                print(f"  {media}: {status}")
    
    # 統計情報
    print(f"\n=== 統計情報 ===")
    print(f"生成人数: {len(personas)}")
    
    ages = [p.age for p in personas]
    print(f"年齢範囲: {min(ages)}歳 - {max(ages)}歳")
    
    generations = [p.generation for p in personas]
    from collections import Counter
    gen_counts = Counter(generations)
    print("世代分布:")
    for gen, count in gen_counts.items():
        print(f"  {gen}: {count}人")
    
    genders = [p.gender for p in personas]
    gender_counts = Counter(genders)
    print("性別分布:")
    for gender, count in gender_counts.items():
        print(f"  {gender}: {count}人")
    
    print("\n✅ ペルソナ生成テスト完了")

if __name__ == "__main__":
    test_persona_generation()