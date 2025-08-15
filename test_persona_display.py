#!/usr/bin/env python3
# テスト用スクリプト：新しいペルソナ表示のテスト

import sys
import os
import pandas as pd
sys.path.append(os.path.dirname(__file__))

from app import JapanDemographicsDB, PersonaGenerator

def test_persona_display():
    print("=== 新しいペルソナ表示テスト ===")
    
    # データベースとジェネレーターを初期化
    demographics_db = JapanDemographicsDB()
    persona_generator = PersonaGenerator(demographics_db)
    
    # 10人のペルソナを生成
    personas = []
    for i in range(10):
        persona = persona_generator.generate_persona(i + 1)
        persona_dict = {
            'id': persona.id,
            'age': persona.age,
            'gender': persona.gender,
            'occupation': persona.occupation,
            'education': persona.education,
            'income_level': persona.income_level,
            'family_status': persona.family_status,
            'media_preferences': persona.media_preferences
        }
        personas.append(persona_dict)
    
    df = pd.DataFrame(personas)
    
    print(f"\n📊 基本統計:")
    print(f"生成済みペルソナ数: {len(personas)}")
    print(f"平均年齢: {df['age'].mean():.1f}歳")
    
    print(f"\n📈 就職氷河期世代の雇用状況:")
    occupation_counts = df['occupation'].value_counts()
    for occupation, count in occupation_counts.items():
        percentage = (count / len(personas)) * 100
        print(f"  {occupation}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n🎓 学歴分布:")
    education_counts = df['education'].value_counts()
    for education, count in education_counts.items():
        percentage = (count / len(personas)) * 100
        print(f"  {education}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n💰 収入分布:")
    income_counts = df['income_level'].value_counts()
    for income, count in income_counts.items():
        percentage = (count / len(personas)) * 100
        print(f"  {income}: {count}人 ({percentage:.1f}%)")
    
    print(f"\n👨‍👩‍👧‍👦 家族構成分布:")
    family_counts = df['family_status'].value_counts()
    for family, count in family_counts.items():
        percentage = (count / len(personas)) * 100
        print(f"  {family}: {count}人 ({percentage:.1f}%)")
    
    # 重要な統計指標
    non_regular_ratio = (df['occupation'].isin(['不本意非正規', '失業・無業'])).mean()
    print(f"\n📊 重要指標:")
    print(f"非正規・無業比率: {non_regular_ratio:.1%}")
    
    high_income_ratio = (df['income_level'].isin(['600-800万円', '800-1000万円', '1000万円以上'])).mean()
    print(f"中高収入層比率: {high_income_ratio:.1%}")
    
    print(f"\n📋 ペルソナ詳細サンプル（最初の5人）:")
    display_df = df[['id', 'age', 'gender', 'occupation', 'education', 'income_level', 'family_status']].head(5).copy()
    display_df.columns = ['ID', '年齢', '性別', '職業', '学歴', '収入レベル', '家族構成']
    print(display_df.to_string(index=False))
    
    # メディア接触パターンの表示
    print(f"\n📺 メディア接触パターン（サンプル）:")
    media_names = {
        'instagram': 'Instagram', 'x': 'X', 'youtube': 'YouTube',
        'tv': 'TV', 'newspaper_print': '新聞(紙)', 
        'newspaper_digital': '新聞(デジタル)', 'transit_ads': '交通広告'
    }
    
    for i, persona in enumerate(personas[:3], 1):
        contacts = []
        if persona.get('media_preferences'):
            for media, contact in persona['media_preferences'].items():
                if contact and media in media_names:
                    contacts.append(media_names[media])
        print(f"  ペルソナ{i}（{persona['age']}歳 {persona['gender']}）: {', '.join(contacts[:4])}")
    
    print("\n✅ 新しいペルソナ表示テスト完了")

if __name__ == "__main__":
    test_persona_display()