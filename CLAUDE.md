# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Japanese opinion polling simulator called "LLM100人に聞きました" (LLM Asked 100 People). It generates realistic personas based on Japanese demographics and simulates survey responses using either GPT-4o-mini or built-in simulation patterns. The application produces comprehensive PDF reports with AI analysis.

## Architecture

### Core Components

**Demographics & Persona System** (`JapanDemographicsDB`, `PersonaGenerator`, `PersonaProfile`):
- Uses real Japanese demographic data from government sources (総務省人口推計, 就業構造基本調査)
- Generates statistically accurate personas with age, gender, occupation, political leanings, etc.
- Automatically adjusts political tendencies based on age demographics

**LLM Providers** (`GPT4OMiniProvider`, `SimulationProvider`):
- Dual-mode system: real GPT-4o-mini API calls vs. built-in response simulation
- `GPT4OMiniProvider`: Async OpenAI API calls with cost tracking, token limiting (100 chars per response)
- `SimulationProvider`: Pattern-based responses using predefined templates by generation and sentiment

**Analysis Engine** (`ResponseAnalyzer`, `EnhancedPromptGenerator`):
- Keyword extraction with Japanese text processing
- Sentiment analysis using positive/negative word matching
- AI-powered analysis generating 2400-character detailed reports covering social dynamics, generational conflicts, policy implications

**Web Integration** (`WebSearchProvider`):
- DuckDuckGo integration for real-time information retrieval
- Automatic search result summarization to provide context for responses

**Report Generation** (`PDFReportGenerator`):
- Comprehensive PDF reports using ReportLab with Japanese font support
- Includes survey overview, demographics, AI analysis, response samples, keyword analysis, sentiment breakdown

### Application Flow

1. **Persona Generation**: Creates statistically representative sample of Japanese citizens
2. **Survey Execution**: Gathers responses using either GPT-4o-mini or simulation patterns, optionally incorporating web search context
3. **Analysis**: Processes responses through keyword extraction, sentiment analysis, and AI-powered deep analysis
4. **Report Generation**: Creates downloadable PDF reports and CSV/JSON data exports

## Development Commands

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Quick setup using included installer
python install.py
```

### Running the Application
```bash
# Start Streamlit development server
streamlit run app.py

# Access at http://localhost:8501
```

### Jupyter Notebook Support
```bash
# Start Jupyter Lab for data analysis
jupyter lab

# Work with data files in notebook/data/
```

## Key Dependencies

**Required for core functionality**:
- `streamlit>=1.28.0` - Web interface
- `pandas>=1.5.0`, `numpy>=1.24.0` - Data processing
- `plotly>=5.15.0` - Visualizations

**Optional for enhanced features**:
- `openai>=1.0.0` - GPT-4o-mini integration
- `reportlab==4.0.4` - PDF generation
- `duckduckgo-search>=3.9.0` - Web search
- `tiktoken>=0.5.0` - Token counting

## Configuration

### API Keys
Set OpenAI API key via:
- Streamlit sidebar input
- Environment variable: `OPENAI_API_KEY`

### Cost Management
- GPT-4o-mini: ~1.2 yen per 100 responses
- AI analysis: ~1.8 yen per analysis
- Built-in cost tracking with USD/JPY conversion

## Testing and Quality

### Dependency Verification
```bash
# Run installation checker with details
python install.py --verbose

# Check library compatibility
python -c "import streamlit, pandas, plotly; print('Core libraries OK')"
```

### Data Validation
The persona generation uses weighted random sampling based on official Japanese statistics. Verify demographic distributions match expected government data in `JapanDemographicsDB.setup_demographics_data()`.

## Code Organization Notes

- **Monolithic Design**: Single `app.py` file contains all functionality for easier deployment
- **Async Operations**: Survey execution uses asyncio for concurrent API calls
- **Error Handling**: Graceful degradation when optional dependencies missing
- **Japanese Localization**: All UI text, demographic data, and analysis in Japanese
- **State Management**: Heavy use of Streamlit session state for multi-tab workflow

## Data Files

- `notebook/data/data.ipynb` - Jupyter notebook for data analysis
- `notebook/data/*.csv` - Survey data files (sy24rv10rc.csv, sy24rv20rc.csv)
- `install_log.txt` - Generated installation verification log