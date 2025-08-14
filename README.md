# 🧠 Decision Room - AI Advisory Board

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://decision-room.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 🚀 Transform Every Decision Into a Board Meeting with AI Advisors

Decision Room is an innovative AI-powered application that provides multi-perspective analysis for life's important decisions. By leveraging CrewAI's multi-agent framework and OpenAI's language models, it simulates a personal advisory board tailored to your specific decision context.

### ✨ Live Demo
🔗 [Try Decision Room Now](https://decision-room.streamlit.app)

![Decision Room Demo](https://via.placeholder.com/800x400/4A90E2/FFFFFF?text=Decision+Room+Demo)

## 🎯 Key Features

### 🤖 **Intelligent Multi-Agent System**
- **Smart Agent Selection**: Automatically detects decision type and assigns relevant experts
- **Specialized Advisors**: From Financial Analysts to Life Coaches, each with unique perspectives
- **Adaptive Depth**: Choose between Quick (2 agents), Standard (4 agents), or Deep Analysis (6 agents)

### 💡 **Smart Context Recognition**
- **Auto-Detection**: Identifies decision type (Financial, Career, Education, etc.) from your question
- **Dynamic Fields**: Context inputs automatically adapt to your specific decision type
- **Relevant Questions**: Only asks for information that matters to your decision

### 💰 **Transparent Cost Management**
- **Model Selection**: Choose between GPT-3.5 ($0.01), GPT-4 Turbo ($0.08), or GPT-4 ($0.15)
- **Real-Time Tracking**: See exact token usage and costs
- **Budget Calculator**: Know how many analyses your budget allows
- **Cost Preview**: See estimated cost before running analysis

### 🔒 **Privacy-First Design**
- **Session-Only API Keys**: Never stored or logged
- **No Data Persistence**: Your decisions remain private
- **Anonymous Analytics**: Optional and completely anonymous
- **Local Processing**: All analysis happens in real-time

## 🛠️ Tech Stack

- **Frontend**: Streamlit (Modern Python web framework)
- **AI Orchestration**: CrewAI (Multi-agent framework)
- **Language Models**: OpenAI GPT-3.5/GPT-4
- **Deployment**: Streamlit Cloud (Free hosting)
- **Analytics**: Built-in anonymous tracking

## 📦 Installation

### Prerequisites
- Python 3.8 or higher
- OpenAI API key ([Get one here](https://platform.openai.com/api-keys))

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/decision-room.git
cd decision-room

Install dependencies

bashpip install -r requirements.txt

Run the application

bashstreamlit run app.py

Open in browser

http://localhost:8501
🎮 Usage Guide
Step 1: Enter Your OpenAI API Key

Add your API key in the sidebar (session-only, never saved)
Don't have one? Get it here

Step 2: Choose Your Settings

Select Model: GPT-3.5 (cheap) or GPT-4 (quality)
Pick Depth: Quick, Standard, or Deep analysis
See Cost: Preview cost before analyzing

Step 3: Describe Your Decision

Type your decision question
Watch as context fields adapt automatically
Add relevant details in dynamic fields

Step 4: Get Multi-Perspective Analysis

Receive insights from specialized AI advisors
See consensus score and recommendations
Review detailed cost breakdown

📊 Decision Types Supported
TypeDescriptionSpecialized Advisors💰 FinancialInvestment, purchases, money decisionsFinancial Advisor, Risk Analyst💼 CareerJob changes, promotions, work decisionsCareer Coach, Industry Expert🎓 EducationLearning, degrees, skill developmentEducation Consultant, ROI Analyst💑 RelationshipPersonal relationships, life changesLife Coach, Psychology Expert🌍 LocationMoving, relocation, travelRelocation Expert, Lifestyle Advisor💻 TechnicalTechnology choices, architectureTechnical Architect, Innovation Scout🎯 GeneralAny other decision typeStrategic Advisor, Opportunity Scout
💡 Example Decisions

"Should I quit my job to start a startup?"
"Should I buy a house now or wait for the market?"
"Should I get an MBA or continue gaining experience?"
"Should I relocate to another country for work?"
"Should I invest in stocks or cryptocurrency?"

🚀 Deployment
Deploy to Streamlit Cloud (Free)

Fork this repository
Connect your GitHub to Streamlit Cloud
Deploy with one click
Share your custom URL

Environment Variables
No environment variables needed! Users provide their own API keys through the UI.
📈 Performance & Costs
ModelCost per AnalysisTokensQualityGPT-3.5 Turbo~$0.01~3,000Good for most decisionsGPT-4 Turbo~$0.08~3,000Better reasoningGPT-4~$0.15~3,000Best quality
Budget Calculator: $1 gets you approximately:

100 analyses with GPT-3.5
12 analyses with GPT-4 Turbo
6 analyses with GPT-4

🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request. For major changes:

Fork the repository
Create your feature branch (git checkout -b feature/AmazingFeature)
Commit your changes (git commit -m 'Add some AmazingFeature')
Push to the branch (git push origin feature/AmazingFeature)
Open a Pull Request

📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

CrewAI - For the amazing multi-agent framework
OpenAI - For GPT models
Streamlit - For the incredible web framework
Community - For feedback and contributions

📞 Contact & Support

Issues: GitHub Issues
Discussions: GitHub Discussions
LinkedIn: Connect with me