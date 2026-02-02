#!/usr/bin/env python3
"""
Fix for structured output JSON parsing issues.

The LLM sometimes returns the schema definition instead of actual values.
This script updates the prompts to be more explicit about the expected format.

Run: python fix_structured_output.py
"""

import re

def fix_llm_gateway():
    """Fix the complete_with_structured_output method in llm_gateway.py"""
    
    filepath = "src/models/llm_gateway.py"
    
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Find and replace the structured output instruction
    old_instruction = '''Respond with a JSON object matching this schema:'''
    
    new_instruction = '''You must respond with ONLY a valid JSON object containing actual values (not a schema definition).

IMPORTANT: Return actual data values, NOT a schema or type definitions.

The JSON object must have these exact fields with real values:'''

    if old_instruction in content:
        content = content.replace(old_instruction, new_instruction)
        print(f"✅ Updated JSON instruction in {filepath}")
    else:
        print(f"⚠️  Could not find old instruction pattern in {filepath}")
        print("   Manual update may be needed")
    
    # Also improve the JSON extraction to handle markdown code blocks better
    old_extraction = '''# Try to extract JSON from the response
            content = response.content
            
            # Handle markdown code blocks
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0]
            elif "```" in content:
                content = content.split("```")[1].split("```")[0]'''
    
    new_extraction = '''# Try to extract JSON from the response
            content = response.content
            
            # Handle markdown code blocks
            if "```json" in content:
                # Extract content between ```json and ```
                match = re.search(r'```json\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)
            elif "```" in content:
                # Extract content between ``` and ```
                match = re.search(r'```\s*([\s\S]*?)\s*```', content)
                if match:
                    content = match.group(1)
            
            # Try to find JSON object in the content
            # Look for content between first { and last }
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                content = content[start:end]'''
    
    if old_extraction in content:
        content = content.replace(old_extraction, new_extraction)
        print(f"✅ Updated JSON extraction in {filepath}")
    
    # Add import for re if not present
    if "import re" not in content and "from re import" not in content:
        # Add after other imports
        content = content.replace(
            "import json",
            "import json\nimport re"
        )
        print(f"✅ Added 're' import")
    
    with open(filepath, 'w') as f:
        f.write(content)
    
    print(f"\n✅ Fixed {filepath}")


def fix_agent_prompts():
    """Update agent prompts to be more explicit about JSON format."""
    
    agents_to_fix = [
        ("src/agents/monitor_agent.py", "AlertClassification"),
        ("src/agents/diagnose_agent.py", "DiagnosisResult"),
        ("src/agents/policy_agent.py", "BlastRadiusAnalysis"),
        ("src/agents/report_agent.py", "IncidentReport"),
    ]
    
    for filepath, model_name in agents_to_fix:
        try:
            with open(filepath, 'r') as f:
                content = f.read()
            
            # Add explicit JSON instruction to system prompts
            # Find patterns like "Be precise and concise" and add JSON instruction after
            
            old_patterns = [
                "Be precise and concise in your analysis.",
                "Be precise, technical, and actionable.",
                "Use professional language. Be specific and actionable.",
            ]
            
            json_instruction = """

CRITICAL: Your response must be a valid JSON object with actual values, not a schema.
Example format (use real values based on your analysis):
"""
            
            for old_pattern in old_patterns:
                if old_pattern in content and json_instruction not in content:
                    content = content.replace(
                        old_pattern,
                        old_pattern + json_instruction
                    )
                    print(f"✅ Updated prompt in {filepath}")
                    break
            
            with open(filepath, 'w') as f:
                f.write(content)
                
        except FileNotFoundError:
            print(f"⚠️  File not found: {filepath}")


def add_json_example_to_diagnose():
    """Add explicit JSON example to diagnose agent."""
    
    filepath = "src/agents/diagnose_agent.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        # Find the user prompt and add JSON example
        old_text = "Based on all this information, provide your diagnosis and recommended actions."
        
        new_text = '''Based on all this information, provide your diagnosis and recommended actions.

Your response MUST be a valid JSON object with these exact fields (use actual values, not descriptions):
{
  "root_cause": "string describing the actual root cause you identified",
  "confidence": 0.85,
  "reasoning": "string with your step-by-step reasoning",
  "memory_informed": true or false,
  "recommended_actions": [
    {
      "action_type": "scale",
      "command": "kubectl scale deployment/api --replicas=5 -n api",
      "target_resource": "deployment/api",
      "description": "Scale up to handle load",
      "risk_level": "low",
      "estimated_impact": "Increased capacity",
      "rollback_command": "kubectl scale deployment/api --replicas=3 -n api"
    }
  ]
}'''
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"✅ Added JSON example to {filepath}")
        else:
            print(f"⚠️  Could not find target text in {filepath}")
            
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")


def add_json_example_to_monitor():
    """Add explicit JSON example to monitor agent."""
    
    filepath = "src/agents/monitor_agent.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        old_text = "Classify this alert and identify all anomalies and affected resources."
        
        new_text = '''Classify this alert and identify all anomalies and affected resources.

Your response MUST be a valid JSON object with these exact fields:
{
  "severity": "P2",
  "anomalies": ["High CPU usage at 85%", "Pod restarts detected"],
  "affected_resources": ["deployment/api-server", "pod/api-server-xyz"],
  "summary": "Brief summary of the alert",
  "confidence": 0.9
}'''
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"✅ Added JSON example to {filepath}")
            
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")


def add_json_example_to_policy():
    """Add explicit JSON example to policy agent."""
    
    filepath = "src/agents/policy_agent.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        old_text = "Provide your blast radius analysis."
        
        new_text = '''Provide your blast radius analysis.

Your response MUST be a valid JSON object with these exact fields:
{
  "score": 0.7,
  "affected_pods_estimate": 10,
  "affected_services": ["api-service", "web-service"],
  "risk_factors": ["Production namespace", "Peak traffic hours"],
  "recommendation": "Proceed with caution, monitor closely"
}'''
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"✅ Added JSON example to {filepath}")
            
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")


def add_json_example_to_report():
    """Add explicit JSON example to report agent."""
    
    filepath = "src/agents/report_agent.py"
    
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        old_text = "Generate the incident report."
        
        new_text = '''Generate the incident report.

Your response MUST be a valid JSON object with these exact fields:
{
  "title": "High CPU Usage Incident Report",
  "summary": "Brief executive summary",
  "root_cause_analysis": "Detailed technical analysis",
  "actions_taken": ["Scaled deployment", "Adjusted resource limits"],
  "outcome": "Issue resolved, service restored",
  "lessons_learned": ["Monitor resource usage more closely"],
  "prevention_recommendations": ["Implement auto-scaling"],
  "slack_message": "🟢 *INC-123 Resolved* - High CPU usage fixed by scaling"
}'''
        
        if old_text in content:
            content = content.replace(old_text, new_text)
            
            with open(filepath, 'w') as f:
                f.write(content)
            
            print(f"✅ Added JSON example to {filepath}")
            
    except FileNotFoundError:
        print(f"⚠️  File not found: {filepath}")


if __name__ == "__main__":
    print("🔧 Fixing structured output JSON parsing issues...\n")
    
    fix_llm_gateway()
    print()
    
    add_json_example_to_monitor()
    add_json_example_to_diagnose()
    add_json_example_to_policy()
    add_json_example_to_report()
    
    print("\n" + "="*50)
    print("✅ All fixes applied!")
    print("="*50)
    print("\nNow run the simulation again:")
    print("  python scripts/simulate_incident.py --severity P3 --type high_cpu")