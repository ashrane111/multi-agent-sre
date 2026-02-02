import re

# Files to update
files_to_update = [
    "src/models/llm_gateway.py",
    "src/models/router.py",
    "src/config.py",
    "src/cli.py",
    ".env.example",
]

# Replacements
replacements = {
    # Groq - use newer model
    "llama-3.1-70b-versatile": "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant": "llama-3.1-8b-instant",  # This one still works
    # Gemini - use the one that worked
    "gemini-1.5-flash": "gemini-flash-latest",
    "gemini-2.0-flash": "gemini-flash-latest",
}

for filepath in files_to_update:
    try:
        with open(filepath, 'r') as f:
            content = f.read()
        
        original = content
        for old, new in replacements.items():
            content = content.replace(old, new)
        
        if content != original:
            with open(filepath, 'w') as f:
                f.write(content)
            print(f"✅ Updated: {filepath}")
        else:
            print(f"⏭️  No changes: {filepath}")
    except FileNotFoundError:
        print(f"❌ Not found: {filepath}")

print("\nDone! Now update your .env file too.")
