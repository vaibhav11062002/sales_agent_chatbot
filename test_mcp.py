from data_connector import mcp_store
import json

def test_mcp_store():
    """Test MCP-backed data store"""
    
    print("🧪 Testing MCP Data Store...\n")
    
    # Test 1: Load data
    print("1️⃣ Loading sales data...")
    try:
        mcp_store.load_sales_data()
        df = mcp_store.get_sales_data()
        print(f"   ✅ Loaded {len(df)} records, {len(df.columns)} columns\n")
    except Exception as e:
        print(f"   ❌ Load failed: {e}\n")
        return
    
    # Test 2: Get data summary
    print("2️⃣ Reading data summary...")
    try:
        df = mcp_store.get_sales_data()
        summary = {
            "total_records": len(df),
            "total_sales": float(df['NetAmount'].sum()) if 'NetAmount' in df.columns else 0,
            "unique_customers": int(df['SoldToParty'].nunique()) if 'SoldToParty' in df.columns else 0,
        }
        print(f"   Summary: {json.dumps(summary, indent=2)}\n")
    except Exception as e:
        print(f"   ❌ Summary failed: {e}\n")
    
    # Test 3: Update agent context
    print("3️⃣ Updating agent context...")
    try:
        mcp_store.update_agent_context("TestAgent", {
            "test": "data",
            "result": 12345
        })
        print(f"   ✅ Context updated\n")
    except Exception as e:
        print(f"   ❌ Update failed: {e}\n")
    
    # Test 4: Read agent context
    print("4️⃣ Reading agent context...")
    try:
        context = mcp_store.get_agent_context("TestAgent")
        print(f"   Context: {json.dumps(context, indent=2)}\n")
    except Exception as e:
        print(f"   ❌ Context read failed: {e}\n")
    
    # Test 5: Read all contexts
    print("5️⃣ Reading all contexts...")
    try:
        all_contexts = mcp_store.get_all_contexts()
        print(f"   All Contexts: {json.dumps(all_contexts, indent=2)}\n")
    except Exception as e:
        print(f"   ❌ All contexts failed: {e}\n")
    
    # Test 6: Conversation history
    print("6️⃣ Testing conversation history...")
    try:
        mcp_store.add_conversation_turn("user", "What are total sales?")
        mcp_store.add_conversation_turn("assistant", "Total sales: $1,000,000")
        print(f"   ✅ {len(mcp_store.conversation_history)} turns in history\n")
    except Exception as e:
        print(f"   ❌ History failed: {e}\n")
    
    print("✅ All Tests Complete!")

if __name__ == "__main__":
    test_mcp_store()
