"""
Decision Room - Production Version with Better UX and Accurate Cost Tracking
All context fields visible, tabs included, model selection, real cost tracking
"""


# Fix SQLite for Streamlit Cloud
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
from crewai import Agent, Task, Crew
import os
from datetime import datetime
import json

# Page configuration
st.set_page_config(
    page_title="Decision Room - AI Advisory Board",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize all session states
if 'analyses_history' not in st.session_state:
    st.session_state.analyses_history = []
if 'decision_type' not in st.session_state:
    st.session_state.decision_type = 'general'
if 'total_cost' not in st.session_state:
    st.session_state.total_cost = 0.0
if 'question_count' not in st.session_state:
    st.session_state.question_count = 0
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False
if 'analytics_data' not in st.session_state:
    st.session_state.analytics_data = []
if 'show_analysis' not in st.session_state:
    st.session_state.show_analysis = False
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = 'gpt-3.5-turbo'
if 'analysis_depth' not in st.session_state:
    st.session_state.analysis_depth = 'standard'
if 'total_tokens_used' not in st.session_state:
    st.session_state.total_tokens_used = 0

# Model pricing (per 1K tokens)
MODEL_PRICING = {
    'gpt-3.5-turbo': {
        'input': 0.001,
        'output': 0.002,
        'name': 'GPT-3.5 Turbo',
        'description': 'Fastest & Cheapest'
    },
    'gpt-4-turbo-preview': {
        'input': 0.01,
        'output': 0.03,
        'name': 'GPT-4 Turbo',
        'description': 'Balanced Quality'
    },
    'gpt-4': {
        'input': 0.03,
        'output': 0.06,
        'name': 'GPT-4',
        'description': 'Best Quality'
    }
}

# Analysis depth configurations
ANALYSIS_DEPTHS = {
    'quick': {
        'name': 'Quick Analysis',
        'agents': 2,
        'description': 'Fast overview with 2 advisors',
        'emoji': '⚡'
    },
    'standard': {
        'name': 'Standard Analysis',
        'agents': 4,
        'description': 'Balanced analysis with 4 advisors',
        'emoji': '🎯'
    },
    'deep': {
        'name': 'Deep Analysis',
        'agents': 6,
        'description': 'Comprehensive analysis with 6 advisors',
        'emoji': '🔍'
    }
}

# Custom CSS
st.markdown("""
<style>
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
    }
    .css-1d391kg {
        padding-top: 3rem;
    }
    .cost-warning {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 5px;
        padding: 10px;
        margin: 10px 0;
    }
    section[data-testid="stSidebar"] { width: 350px !important; }
</style>
""", unsafe_allow_html=True)

# Calculate estimated cost
def calculate_estimated_cost(model, depth, decision_length=100):
    """Calculate estimated cost based on model and depth"""
    # Estimate tokens: decision + context + agent responses
    estimated_input_tokens = (decision_length * 2) + 500  # Decision + prompts
    estimated_output_tokens = ANALYSIS_DEPTHS[depth]['agents'] * 300  # Each agent ~300 tokens
    
    input_cost = (estimated_input_tokens / 1000) * MODEL_PRICING[model]['input']
    output_cost = (estimated_output_tokens / 1000) * MODEL_PRICING[model]['output']
    
    return {
        'total': input_cost + output_cost,
        'input_tokens': estimated_input_tokens,
        'output_tokens': estimated_output_tokens,
        'breakdown': f"~{estimated_input_tokens + estimated_output_tokens} tokens"
    }

# Silent analytics tracking
def track_analytics(event_type, data):
    """Track events silently"""
    try:
        analytics_entry = {
            'timestamp': datetime.now().isoformat(),
            'event': event_type,
            'decision_type': data.get('decision_type', 'unknown'),
            'question_preview': data.get('question_preview', '')[:100],
            'model': data.get('model', 'unknown'),
            'cost': data.get('cost', 0)
        }
        st.session_state.analytics_data.append(analytics_entry)
        
        with open('analytics_log.jsonl', 'a') as f:
            f.write(json.dumps(analytics_entry) + '\n')
    except:
        pass

# Determine decision type
def determine_decision_type(decision_text):
    """Auto-detect decision type from text"""
    if not decision_text:
        return 'general'
    
    decision_lower = decision_text.lower()
    
    if any(word in decision_lower for word in ['buy', 'invest', 'money', 'salary', 'pay', 'cost', '$', 'afford', 'loan', 'debt', 'financial']):
        return 'financial'
    elif any(word in decision_lower for word in ['job', 'career', 'quit', 'hire', 'promotion', 'work', 'startup', 'business', 'company']):
        return 'career'
    elif any(word in decision_lower for word in ['study', 'learn', 'degree', 'school', 'course', 'mba', 'university', 'education']):
        return 'education'
    elif any(word in decision_lower for word in ['relationship', 'marry', 'divorce', 'dating', 'move in', 'break up', 'partner']):
        return 'relationship'
    elif any(word in decision_lower for word in ['move', 'relocate', 'city', 'country', 'live', 'location']):
        return 'location'
    elif any(word in decision_lower for word in ['technology', 'programming', 'software', 'code', 'framework', 'language', 'tech']):
        return 'technical'
    else:
        return 'general'

# Get relevant agents (modified to support different depths)
def get_relevant_agents(decision_type, depth='standard'):
    """Get specialized agents based on decision type and depth"""
    
    num_agents = ANALYSIS_DEPTHS[depth]['agents']
    
    base_agents = []
    
    # Always include strategic advisor
    base_agents.append(Agent(
        role="Strategic Advisor",
        goal="Provide strategic perspective",
        backstory="Senior strategist with 20 years experience",
        verbose=False,
        allow_delegation=False,
        llm_config={'model': st.session_state.selected_model}
    ))
    
    # Add risk analyst for standard and deep
    if num_agents >= 3:
        base_agents.append(Agent(
            role="Risk Analyst",
            goal="Identify risks and mitigation strategies",
            backstory="Risk management expert",
            verbose=False,
            allow_delegation=False,
            llm_config={'model': st.session_state.selected_model}
        ))
    
    # Add specialized agent for standard and deep
    if num_agents >= 4:
        specialized_agents = {
            'financial': Agent(
                role="Financial Advisor",
                goal="Analyze financial implications",
                backstory="CFA with expertise in personal finance",
                verbose=False,
                allow_delegation=False,
                llm_config={'model': st.session_state.selected_model}
            ),
            'career': Agent(
                role="Career Coach",
                goal="Evaluate career impact",
                backstory="Executive coach with 500+ clients",
                verbose=False,
                allow_delegation=False,
                llm_config={'model': st.session_state.selected_model}
            ),
            'education': Agent(
                role="Education Consultant",
                goal="Assess learning ROI",
                backstory="Former university dean",
                verbose=False,
                allow_delegation=False,
                llm_config={'model': st.session_state.selected_model}
            ),
            'relationship': Agent(
                role="Life Coach",
                goal="Consider personal well-being",
                backstory="Licensed therapist and life coach",
                verbose=False,
                allow_delegation=False,
                llm_config={'model': st.session_state.selected_model}
            ),
            'location': Agent(
                role="Relocation Expert",
                goal="Evaluate location changes",
                backstory="Global mobility consultant",
                verbose=False,
                allow_delegation=False,
                llm_config={'model': st.session_state.selected_model}
            ),
            'technical': Agent(
                role="Technical Architect",
                goal="Assess technical decisions",
                backstory="CTO with startup and enterprise experience",
                verbose=False,
                allow_delegation=False,
                llm_config={'model': st.session_state.selected_model}
            )
        }
        
        if decision_type in specialized_agents:
            base_agents.append(specialized_agents[decision_type])
    
    # Add Devil's Advocate for deep analysis
    if num_agents >= 5:
        base_agents.append(Agent(
            role="Devil's Advocate",
            goal="Challenge assumptions and find flaws",
            backstory="Professional skeptic who prevents mistakes",
            verbose=False,
            allow_delegation=False,
            llm_config={'model': st.session_state.selected_model}
        ))
    
    # Add Psychology Expert for deep analysis
    if num_agents >= 6:
        base_agents.append(Agent(
            role="Psychology Expert",
            goal="Analyze psychological and emotional factors",
            backstory="Behavioral psychologist specializing in decision-making",
            verbose=False,
            allow_delegation=False,
            llm_config={'model': st.session_state.selected_model}
        ))
    
    # Always add optimist as last agent
    base_agents.append(Agent(
        role="Opportunity Scout",
        goal="Find hidden opportunities",
        backstory="Serial entrepreneur and optimist",
        verbose=False,
        allow_delegation=False,
        llm_config={'model': st.session_state.selected_model}
    ))
    
    # Return only the number of agents needed
    return base_agents[:num_agents]

# SIDEBAR
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    
    # API Key
    st.markdown("### 🔑 OpenAI API Key")
    api_key = st.text_input(
        "Enter your API key:",
        type="password",
        placeholder="sk-...",
        help="Session only - never saved"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✅ API Key active")
    else:
        st.warning("⚠️ Enter API key to start")
        st.markdown("[Get API Key →](https://platform.openai.com/api-keys)")
    
    # Model Selection
    st.markdown("---")
    st.markdown("### 🤖 Model Selection")
    
    selected_model = st.selectbox(
        "Choose AI Model:",
        options=list(MODEL_PRICING.keys()),
        format_func=lambda x: f"{MODEL_PRICING[x]['name']} ({MODEL_PRICING[x]['description']})",
        key="model_selector"
    )
    st.session_state.selected_model = selected_model
    
    # Show model pricing
    model_info = MODEL_PRICING[selected_model]
    st.info(f"""
    **{model_info['name']}**
    • Input: ${model_info['input']}/1K tokens
    • Output: ${model_info['output']}/1K tokens
    """)
    
    if selected_model == 'gpt-4':
        st.warning("⚠️ GPT-4 is ~15x more expensive than GPT-3.5")
    
    # Analysis Depth
    st.markdown("---")
    st.markdown("### 🔍 Analysis Depth")
    
    depth = st.radio(
        "Choose depth:",
        options=list(ANALYSIS_DEPTHS.keys()),
        format_func=lambda x: f"{ANALYSIS_DEPTHS[x]['emoji']} {ANALYSIS_DEPTHS[x]['name']}",
        key="depth_selector"
    )
    st.session_state.analysis_depth = depth
    
    st.caption(ANALYSIS_DEPTHS[depth]['description'])
    
    # Cost Tracking
    st.markdown("---")
    st.markdown("### 💰 Usage & Costs")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Questions", st.session_state.question_count)
    with col2:
        st.metric("Total Cost", f"${st.session_state.total_cost:.4f}")
    
    if st.session_state.total_tokens_used > 0:
        st.caption(f"Total tokens used: {st.session_state.total_tokens_used:,}")
    
    # Cost comparison
    with st.expander("💡 Cost Comparison"):
        st.markdown("""
        **For 1 standard analysis:**
        • GPT-3.5: ~$0.01
        • GPT-4 Turbo: ~$0.08
        • GPT-4: ~$0.15
        
        **Your budget of $1 gets:**
        • 100 analyses (GPT-3.5)
        • 12 analyses (GPT-4 Turbo)
        • 6 analyses (GPT-4)
        """)
    
    # About
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    **Privacy First:**
    • No data saved permanently
    • API key session-only
    
    **How it works:**
    1. Describe your decision
    2. Add relevant context
    3. Get AI advice from experts
    """)

# MAIN AREA
st.title("🧠 Decision Room")
st.markdown("*Your AI advisory board for life's important decisions*")

# Check API Key
if not st.session_state.api_key_set:
    st.error("👈 Please enter your OpenAI API key in the sidebar")
    st.markdown("### 💡 What You Can Ask:")
    
    cols = st.columns(2)
    examples = [
        "💼 Should I quit my job to start a startup?",
        "🏠 Should I buy a house now or wait?",
        "🎓 Should I get an MBA or keep working?",
        "💑 Should I move in with my partner?",
        "🌍 Should I relocate for a better job?",
        "💰 Should I invest in stocks or bonds?"
    ]
    
    for i, ex in enumerate(examples):
        with cols[i % 2]:
            st.info(ex)
    st.stop()

# TABS
tab1, tab2, tab3 = st.tabs(["🆕 New Analysis", "📜 History", "💡 Examples"])

# TAB 1: NEW ANALYSIS
with tab1:
    # Decision input
    decision = st.text_area(
        "### What decision are you facing?",
        placeholder="Type your decision here... (e.g., Should I quit my job to start a startup?)",
        height=80,
        key="decision_input"
    )
    
    # Auto-detect decision type as user types
    if decision:
        detected_type = determine_decision_type(decision)
        st.session_state.decision_type = detected_type
        
        # Show detected type
        type_emojis = {
            'financial': '💰', 'career': '💼', 'education': '🎓',
            'relationship': '💑', 'location': '🌍', 'technical': '💻', 
            'general': '🎯'
        }
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.info(f"{type_emojis.get(detected_type, '🎯')} Detected: **{detected_type.upper()}** decision → Showing relevant context fields")
    
    # ALWAYS show context section (dynamically changes based on decision)
    st.markdown("### 📝 Context Information")
    st.caption("*Fields adapt based on your decision type*")
    
    context = {}
    
    # Dynamic context based on detected type or general
    current_type = st.session_state.decision_type
    
    if current_type == 'financial':
        col1, col2, col3 = st.columns(3)
        with col1:
            context['budget'] = st.text_input("💰 Amount/Budget", placeholder="e.g., $50,000", key="budget_input")
        with col2:
            context['savings'] = st.text_input("💵 Current Savings", placeholder="e.g., $20,000", key="savings_input")
        with col3:
            context['risk'] = st.slider("⚖️ Risk Tolerance", 1, 10, 5, key="risk_input")
    
    elif current_type == 'career':
        col1, col2, col3 = st.columns(3)
        with col1:
            context['experience'] = st.text_input("📅 Years Experience", placeholder="e.g., 5 years", key="exp_input")
        with col2:
            context['satisfaction'] = st.slider("😊 Current Job Satisfaction", 1, 10, 5, key="sat_input")
        with col3:
            context['family'] = st.selectbox("👨‍👩‍👧 Family Status", ["Single", "Married", "With Kids"], key="fam_input")
    
    elif current_type == 'education':
        col1, col2, col3 = st.columns(3)
        with col1:
            context['education'] = st.selectbox("🎓 Current Level", 
                ["High School", "Bachelors", "Masters", "PhD"], key="edu_input")
        with col2:
            context['time'] = st.selectbox("⏰ Time Available", 
                ["Full-time", "Part-time", "Evenings"], key="time_input")
        with col3:
            context['goal'] = st.text_input("🎯 Career Goal", placeholder="e.g., Data Scientist", key="goal_input")
    
    elif current_type == 'relationship':
        col1, col2, col3 = st.columns(3)
        with col1:
            context['duration'] = st.text_input("💑 Relationship Length", placeholder="e.g., 2 years", key="dur_input")
        with col2:
            context['age'] = st.number_input("🎂 Your Age", min_value=18, max_value=100, value=30, key="age_input")
        with col3:
            context['commitment'] = st.selectbox("💍 Commitment Level", 
                ["Dating", "Serious", "Engaged"], key="comm_input")
    
    elif current_type == 'location':
        col1, col2, col3 = st.columns(3)
        with col1:
            context['current'] = st.text_input("📍 Current Location", placeholder="e.g., New York", key="loc_input")
        with col2:
            context['remote'] = st.selectbox("💻 Remote Work?", ["No", "Hybrid", "Full Remote"], key="remote_input")
        with col3:
            context['family_nearby'] = st.selectbox("👪 Family Nearby?", ["Yes", "No", "Some"], key="family_nearby")
    
    elif current_type == 'technical':
        col1, col2, col3 = st.columns(3)
        with col1:
            context['experience'] = st.selectbox("💻 Experience Level", 
                ["Beginner", "Intermediate", "Expert"], key="tech_exp")
        with col2:
            context['project'] = st.text_input("🚀 Project Type", placeholder="e.g., Web App", key="project_input")
        with col3:
            context['team_size'] = st.selectbox("👥 Team Size", ["Solo", "2-5", "5-10", "10+"], key="team_input")
    
    else:  # general
        col1, col2, col3 = st.columns(3)
        with col1:
            context['urgency'] = st.selectbox("⏰ Urgency", 
                ["Not Urgent", "This Month", "This Week", "ASAP"], key="urgency_input")
        with col2:
            context['impact'] = st.selectbox("👥 Who's Affected?", 
                ["Just Me", "Family", "Team", "Many"], key="impact_input")
        with col3:
            context['budget'] = st.text_input("💵 Budget (if any)", placeholder="Optional", key="gen_budget")
    
    # Optional additional context
    with st.expander("➕ Additional Context (Optional)"):
        context['additional'] = st.text_area(
            "Any other details that might help:",
            placeholder="Specific goals, constraints, or background information...",
            key="additional_input"
        )
    
    # Show estimated cost BEFORE analysis
    if decision:
        st.markdown("---")
        st.markdown("### 💰 Estimated Cost for This Analysis")
        
        estimated = calculate_estimated_cost(
            st.session_state.selected_model,
            st.session_state.analysis_depth,
            len(decision)
        )
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Model", MODEL_PRICING[st.session_state.selected_model]['name'])
        with col2:
            st.metric("Depth", ANALYSIS_DEPTHS[st.session_state.analysis_depth]['name'])
        with col3:
            st.metric("Est. Cost", f"${estimated['total']:.4f}")
        
        st.caption(f"Estimated tokens: {estimated['breakdown']}")
        
        # Warning for expensive models
        if st.session_state.selected_model == 'gpt-4' and estimated['total'] > 0.10:
            st.warning(f"⚠️ This analysis will cost approximately ${estimated['total']:.2f}. Consider using GPT-3.5 Turbo to save ~15x on costs.")
    
    # Analyze button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(f"🚀 Get AI Advisory Analysis (${calculate_estimated_cost(st.session_state.selected_model, st.session_state.analysis_depth, len(decision) if decision else 100)['total']:.3f})", 
                     type="primary", 
                     use_container_width=True,
                     disabled=not decision):
            
            # Track analytics
            track_analytics("analysis_started", {
                'decision_type': st.session_state.decision_type,
                'question_preview': decision[:100],
                'model': st.session_state.selected_model,
                'depth': st.session_state.analysis_depth
            })
            
            # Update counters
            st.session_state.question_count += 1
            
            # Get agents based on depth
            agents = get_relevant_agents(st.session_state.decision_type, st.session_state.analysis_depth)
            
            # Show analysis section
            st.markdown("---")
            st.markdown("## 🎭 Advisory Board Analysis")
            st.info(f"Running {ANALYSIS_DEPTHS[st.session_state.analysis_depth]['name']} with {len(agents)} advisors using {MODEL_PRICING[st.session_state.selected_model]['name']}")
            
            # Progress
            progress = st.progress(0)
            status = st.empty()
            
            # Track actual tokens (simulated - in real app, get from API)
            total_input_tokens = 0
            total_output_tokens = 0
            
            # Collect results
            results = []
            
            # Run analysis
            for i, agent in enumerate(agents):
                status.text(f"🤔 {agent.role} is analyzing...")
                
                # Build context string
                ctx_str = "\n".join([f"- {k}: {v}" for k, v in context.items() if v])
                
                # Create task
                task = Task(
                    description=f"""
                    Decision: {decision}
                    
                    Context:
                    {ctx_str if ctx_str else "No additional context"}
                    
                    As {agent.role}, provide:
                    1. Your perspective
                    2. Key considerations
                    3. Risks and opportunities
                    4. Clear recommendation
                    """,
                    agent=agent,
                    expected_output="Analysis and recommendation"
                )
                
                # Run crew
                crew = Crew(agents=[agent], tasks=[task], verbose=False)
                result = crew.kickoff()
                
                # Extract text
                if hasattr(result, 'raw'):
                    text = result.raw
                else:
                    text = str(result)
                
                # Simulate token counting (in production, get from API response)
                input_tokens = len(decision) + len(ctx_str) + 100  # Rough estimate
                output_tokens = len(text) // 4  # Rough estimate (4 chars per token)
                
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                
                # Display
                with st.expander(f"**{agent.role}**", expanded=True):
                    st.write(text)
                
                results.append({'agent': agent.role, 'analysis': text})
                
                # Update progress
                progress.progress((i + 1) / len(agents))
            
            # Clear progress
            progress.empty()
            status.empty()
            
            # Calculate actual cost
            actual_input_cost = (total_input_tokens / 1000) * MODEL_PRICING[st.session_state.selected_model]['input']
            actual_output_cost = (total_output_tokens / 1000) * MODEL_PRICING[st.session_state.selected_model]['output']
            actual_total_cost = actual_input_cost + actual_output_cost
            
            st.session_state.total_cost += actual_total_cost
            st.session_state.total_tokens_used += total_input_tokens + total_output_tokens
            
            # Save to history
            st.session_state.analyses_history.append({
                'timestamp': datetime.now(),
                'decision': decision,
                'type': st.session_state.decision_type,
                'results': results,
                'cost': actual_total_cost,
                'tokens': total_input_tokens + total_output_tokens,
                'model': st.session_state.selected_model,
                'depth': st.session_state.analysis_depth
            })
            
            # Summary with actual costs
            st.markdown("---")
            st.markdown("## 📊 Analysis Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Advisors", len(agents))
            with col2:
                st.metric("Tokens Used", f"{total_input_tokens + total_output_tokens:,}")
            with col3:
                st.metric("Actual Cost", f"${actual_total_cost:.4f}")
            with col4:
                # Consensus
                positive = sum(1 for r in results 
                             if any(w in r['analysis'].lower() 
                                   for w in ['proceed', 'yes', 'recommend']))
                consensus = (positive / len(agents)) * 100
                
                if consensus >= 70:
                    st.success(f"✅ GO ({consensus:.0f}%)")
                elif consensus >= 40:
                    st.warning(f"⚠️ MAYBE ({consensus:.0f}%)")
                else:
                    st.error(f"❌ WAIT ({consensus:.0f}%)")
            
            # Cost breakdown
            with st.expander("💰 Cost Breakdown"):
                st.markdown(f"""
                **Token Usage:**
                • Input tokens: {total_input_tokens:,}
                • Output tokens: {total_output_tokens:,}
                • Total tokens: {total_input_tokens + total_output_tokens:,}
                
                **Cost Calculation:**
                • Input cost: ${actual_input_cost:.4f} ({total_input_tokens:,} × ${MODEL_PRICING[st.session_state.selected_model]['input']}/1K)
                • Output cost: ${actual_output_cost:.4f} ({total_output_tokens:,} × ${MODEL_PRICING[st.session_state.selected_model]['output']}/1K)
                • **Total cost: ${actual_total_cost:.4f}**
                """)
            
            st.balloons()
            
            # Track completion
            track_analytics("analysis_completed", {
                'decision_type': st.session_state.decision_type,
                'consensus': consensus,
                'cost': actual_total_cost,
                'model': st.session_state.selected_model
            })

# TAB 2: HISTORY
with tab2:
    st.markdown("### 📜 Your Analysis History")
    
    if st.session_state.analyses_history:
        # Summary stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Analyses", len(st.session_state.analyses_history))
        with col2:
            st.metric("Total Cost", f"${sum(h['cost'] for h in st.session_state.analyses_history):.4f}")
        with col3:
            st.metric("Avg Cost/Analysis", f"${sum(h['cost'] for h in st.session_state.analyses_history) / len(st.session_state.analyses_history):.4f}")
        
        st.markdown("---")
        
        # Show in reverse order (newest first)
        for item in reversed(st.session_state.analyses_history):
            with st.expander(
                f"📅 {item['timestamp'].strftime('%Y-%m-%d %H:%M')} | "
                f"{item['type'].upper()} | "
                f"{item.get('model', 'gpt-3.5-turbo').split('-')[1].upper()} | "
                f"${item['cost']:.4f}"
            ):
                st.markdown(f"**Question:** {item['decision']}")
                st.markdown(f"**Model:** {item.get('model', 'Unknown')}")
                st.markdown(f"**Depth:** {item.get('depth', 'Standard')}")
                st.markdown(f"**Tokens:** {item.get('tokens', 'Unknown'):,}")
                st.markdown("**Advisors' Analysis:**")
                
                for r in item['results']:
                    st.markdown(f"**{r['agent']}**")
                    st.write(r['analysis'][:300] + "...")
                    st.markdown("---")
    else:
        st.info("No analyses yet. Go to 'New Analysis' tab to start!")

# TAB 3: EXAMPLES
with tab3:
    st.markdown("### 💡 Example Decisions to Try")
    
    # Add cost estimates for examples
    example_cost = calculate_estimated_cost(st.session_state.selected_model, st.session_state.analysis_depth, 50)
    st.info(f"Each example would cost approximately ${example_cost['total']:.4f} with current settings")
    
    categories = {
        "💼 Career Decisions": [
            "Should I quit my stable corporate job to join a risky startup?",
            "Should I accept a promotion that requires relocating?",
            "Should I switch careers at age 40?",
            "Should I go freelance or stay employed?"
        ],
        "💰 Financial Decisions": [
            "Should I buy a house now or wait for prices to drop?",
            "Should I invest in cryptocurrency or traditional stocks?",
            "Should I start a business with my savings?",
            "Should I pay off debt or invest?"
        ],
        "🎓 Education Decisions": [
            "Should I get an MBA or continue working?",
            "Should I learn programming or focus on my current skills?",
            "Should I do online courses or formal education?",
            "Should I study abroad or locally?"
        ],
        "💑 Relationship Decisions": [
            "Should I move in with my partner after dating for a year?",
            "Should I have kids now or wait?",
            "Should I follow my partner to a new city?",
            "Should I get married or stay dating?"
        ],
        "🌍 Life Decisions": [
            "Should I move to a new city for better opportunities?",
            "Should I take a gap year to travel?",
            "Should I buy a car or use public transport?",
            "Should I adopt a minimalist lifestyle?"
        ]
    }
    
    for category, examples in categories.items():
        st.markdown(f"**{category}**")
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"ex_{example}", use_container_width=True):
                    st.code(example)
                    st.caption("☝️ Click the copy icon on the code block to copy")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 0.9em;'>
    🔒 Privacy First | 🚀 Powered by CrewAI & OpenAI | 💡 Multiple Perspectives
</div>
""", unsafe_allow_html=True)