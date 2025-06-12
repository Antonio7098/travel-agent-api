from dotenv import load_dotenv
_ = load_dotenv()

from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, List, Optional
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage, AIMessage
from langchain_google_vertexai import ChatVertexAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import tool
import base64
import json
import time
from langchain_core.runnables import RunnableConfig

model = ChatVertexAI(
    model_name='gemini-2.0-flash-lite-001',
    project='nc-travel-462414',
    location='us-central1',
    max_output_tokens=2048,
    temperature=0.2
)
@tool
def check_trips(firebase_token: str):
    """
    Check trips for a given user by making API requests
    
    Args:
        firebase_token (str): Firebase ID token used to authenticate and extract UID

    Returns:
        dict: JSON response containing trip information
        
    Raises:
        ValueError: If Firebase ID token is missing or invalid
        requests.exceptions.RequestException: If the API request fails
    """
    if not firebase_token:
        raise ValueError("Firebase ID token is required")

    try:
        # Decode Firebase token to get UID
        header, payload, signature = firebase_token.split('.')
        decoded_payload = base64.urlsafe_b64decode(payload + '===').decode()
        payload_data = json.loads(decoded_payload)

        # Check if token is expired
        current_time = int(time.time() * 1000)
        if current_time > payload_data.get('exp', 0) * 1000:
            raise ValueError("Firebase token has expired. Please refresh your token.")

        uid = payload_data.get('user_id')
        if not uid:
            raise ValueError("UID not found in Firebase token")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {firebase_token}"
        }

        response = requests.get(
            "http://localhost:3001/api/trips",
            headers=headers,
            timeout=10
        )

        print(f"Trips response status: {response.status_code}")
        print(f"Response content: {response.text[:200]}..." if len(response.text) > 200 else f"Response content: {response.text}")
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        print(f"Request failed: {error_msg}")
        if hasattr(response, 'status_code') and response.status_code == 403:
            raise ValueError(f"Firebase authentication failed. Response: {error_msg}")
        raise e
    except Exception as e:
        print(f"Unexpected error checking trips: {str(e)}")
        raise e
    
@tool
def check_flights(tripId: str, firebase_token: str):
    """
    Check flights for a specific trip by making API requests
    
    Args:
        tripId (str): ID of the trip to check flights for
        firebase_token (str): Firebase ID token used to authenticate and extract UID

    Returns:
        dict: JSON response containing flight information
        
    Raises:
        ValueError: If tripId or token is invalid
        requests.exceptions.RequestException: If the API request fails
    """
    if not tripId:
        raise ValueError("tripId is required")
    if not firebase_token:
        raise ValueError("Firebase ID token is required")

    try:
        # Decode Firebase token to get UID
        header, payload, signature = firebase_token.split('.')
        decoded_payload = base64.urlsafe_b64decode(payload + '===').decode()
        payload_data = json.loads(decoded_payload)
        uid = payload_data.get('user_id')  # This is the actual UID in the token

        if not uid:
            raise ValueError("UID not found in Firebase token")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {firebase_token}"
        }

        response = requests.get(
            f"http://localhost:3001/api/{tripId}/flights",
            headers=headers,
            json={"user": {"uid": uid}},  # Use dynamic UID
            timeout=10
        )

        print(f"Flights response status: {response.status_code}")
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        print(f"Error checking flights for trip {tripId}: {error_msg}")
        if hasattr(response, 'status_code') and response.status_code == 403:
            raise ValueError(f"Firebase authentication failed. Response: {error_msg}")
        raise e
    except Exception as e:
        print(f"Unexpected error checking flights: {str(e)}")
        raise e
@tool
def check_accom(tripId: str, firebase_token: str):
    """
    Check accommodations for a specific trip by making API requests
    
    Args:
        tripId (str): ID of the trip to check accommodations for
        firebase_token (str): Firebase ID token used to authenticate and extract UID

    Returns:
        dict: JSON response containing accommodation information
        
    Raises:
        ValueError: If tripId or token is invalid
        requests.exceptions.RequestException: If the API request fails
    """
    if not tripId:
        raise ValueError("tripId is required")
    if not firebase_token:
        raise ValueError("Firebase ID token is required")

    try:
        # Decode Firebase token to get UID
        header, payload, signature = firebase_token.split('.')
        decoded_payload = base64.urlsafe_b64decode(payload + '===').decode()
        payload_data = json.loads(decoded_payload)
        uid = payload_data.get('user_id')

        if not uid:
            raise ValueError("UID not found in Firebase token")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {firebase_token}"
        }

        response = requests.get(
            f"http://localhost:3001/api/{tripId}/accommodations",
            headers=headers,
            json={"user": {"uid": uid}},  # Use dynamic UID
            timeout=10
        )

        print(f"Accommodation response status: {response.status_code}")
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        print(f"Error checking accommodations for trip {tripId}: {error_msg}")
        if hasattr(response, 'status_code') and response.status_code == 403:
            raise ValueError(f"Firebase authentication failed. Response: {error_msg}")
        raise e
    except Exception as e:
        print(f"Unexpected error checking accommodations: {str(e)}")
        raise e

@tool
def check_activities(tripId: str, firebase_token: str):
    """
    Check activities for a specific trip by making API requests
    
    Args:
        tripId (str): ID of the trip to check activities for
        firebase_token (str): Firebase ID token used to authenticate and extract UID

    Returns:
        dict: JSON response containing activity information
        
    Raises:
        ValueError: If tripId or token is invalid
        requests.exceptions.RequestException: If the API request fails
    """
    if not tripId:
        raise ValueError("tripId is required")
    if not firebase_token:
        raise ValueError("Firebase ID token is required")

    try:
        # Decode Firebase token to get UID
        header, payload, signature = firebase_token.split('.')
        decoded_payload = base64.urlsafe_b64decode(payload + '===').decode()
        payload_data = json.loads(decoded_payload)
        uid = payload_data.get('user_id')

        if not uid:
            raise ValueError("UID not found in Firebase token")

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {firebase_token}"
        }

        response = requests.get(
            f"http://localhost:3001/api/{tripId}/activities",
            headers=headers,
            json={"user": {"uid": uid}},  # Use dynamic UID
            timeout=10
        )

        print(f"Activities response status: {response.status_code}")
        response.raise_for_status()
        return response.json()

    except requests.exceptions.RequestException as e:
        error_msg = str(e)
        print(f"Error checking activities for trip {tripId}: {error_msg}")
        if hasattr(response, 'status_code') and response.status_code == 403:
            raise ValueError(f"Firebase authentication failed. Response: {error_msg}")
        raise e
    except Exception as e:
        print(f"Unexpected error checking activities: {str(e)}")
        raise e


# class AgentState(TypedDict):
#     messages: Annotated[List[AnyMessage], operator.add]
#     firebase_token: Optional[str]

# class Agent:
#     def __init__(self, model, tools, system=""):
#         self.system = system
#         graph = StateGraph(AgentState)
#         graph.add_node("llm", self.call_gemini)
#         graph.add_node("action", self.take_action)
#         graph.add_conditional_edges(
#             "llm",
#             self.exists_action,
#             {"action": "action", END: END}
#         )
#         graph.add_edge("action", "llm")
#         graph.set_entry_point("llm")
#         self.runnable = graph.compile()
#         self.tools = {t.name: t for t in tools}
#         self.model = model.bind_tools(tools)

#     def exists_action(self, state: AgentState):
#         print("\n=== exists_action ===")
#         result = state["messages"][-1]
#         has_tool_calls = hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
        
#         print(f"Last message type: {type(result).__name__}")
#         print(f"Has tool calls: {has_tool_calls}")
#         if has_tool_calls:
#             print(f"Tool calls: {result.tool_calls}")
        
#         decision = "action" if has_tool_calls else END
#         print(f"Decision: {decision}")
#         return decision
    
#     def call_gemini(self, state: AgentState):
#         print("\n=== call_gemini ===")
#         print(f"Current state: {state}")
        
#         # DEBUG: Check firebase_token in the incoming state
#         firebase_token_in_state = state.get('firebase_token')
#         print(f"🔍 DEBUG: firebase_token in call_gemini state: '{firebase_token_in_state}' (type: {type(firebase_token_in_state)})")
        
#         messages = state['messages']
#         print(f"Input messages ({len(messages)}):")
#         for i, msg in enumerate(messages):
#             print(f"  {i}. {type(msg).__name__}: {msg.content[:200]}..." 
#                 f" (type: {msg.type}, additional_kwargs: {getattr(msg, 'additional_kwargs', {})})")
        
#         if self.system and not any(isinstance(m, SystemMessage) for m in messages):
#             system_msg = SystemMessage(content=self.system)
#             print(f"Adding system message: {self.system[:200]}...")
#             messages = [system_msg] + messages
        
#         print("\nInvoking model...")
#         try:
#             message = self.model.invoke(messages)
#             print(f"Model response: {message.content[:200]}...")
#             if hasattr(message, 'tool_calls'):
#                 print(f"Tool calls: {message.tool_calls}")
            
#             # IMPORTANT: Preserve the firebase_token in the returned state
#             return {
#                 'messages': [message],
#                 'firebase_token': state.get('firebase_token')  # Preserve the firebase_token
#             }
#         except Exception as e:
#             print(f"❌ Error invoking model: {str(e)}")
#             import traceback
#             traceback.print_exc()
#             raise
    
#     def take_action(self, state: AgentState):
#         print("\n=== take_action ===")
#         print(f"State received: {state}")
        
#         # DEBUG: Check what's in the state
#         firebase_token_in_state = state.get('firebase_token')
#         print(f"🔍 DEBUG: firebase_token in state: '{firebase_token_in_state}' (type: {type(firebase_token_in_state)})")
        
#         tool_calls = state['messages'][-1].tool_calls
#         print(f"Tool calls: {tool_calls}")
#         results = []

#         # Get the firebase_token from the state once before the loop
#         firebase_token = state.get('firebase_token')

#         for t in tool_calls:
#             print(f"\nProcessing tool call: {t}")
#             print(f"Tool name: {t['name']}")
            
#             # Copy args from the tool_call
#             tool_args = dict(t['args'])
            
#             # --- START OF CHANGE ---
#             # Always inject the firebase_token from the state.
#             # This is more secure, less ambiguous, and will overwrite any
#             # placeholder value provided by the model.
#             tool_args['firebase_token'] = firebase_token
#             # --- END OF CHANGE ---

#             print(f"✅ Found tool: {t['name']}")
#             print(f"Invoking tool {t['name']} with args: {tool_args}")
            
#             if t['name'] not in self.tools:
#                 print(f"❌ Tool {t['name']} not found")
#                 result = f"Error: Tool '{t['name']}' not found"
#             else:
#                 try:
#                     # The .invoke method will now receive the correct token
#                     result = self.tools[t['name']].invoke(tool_args)
#                     print(f"✅ Tool {t['name']} executed successfully")
#                 except Exception as e:
#                     print(f"❌ Error invoking tool {t['name']}: {e}")
#                     import traceback
#                     traceback.print_exc()
#                     result = f"Error invoking tool {t['name']}: {str(e)}"

#             # Create tool message with the result
#             tool_message = ToolMessage(
#                 tool_call_id=t['id'], 
#                 name=t['name'], 
#                 content=str(result)
#             )
#             print(f"Created ToolMessage: {tool_message}")
#             results.append(tool_message)

#         print(f"\n=== End of take_action ===")
#         print(f"Returning {len(results)} tool messages")
        
#         # Return the results and preserve the token for the next state
#         return {
#             'messages': results,
#             'firebase_token': firebase_token
#         }

# --- CHANGE 1: Remove firebase_token from the state definition ---
class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]
    # The token will now be passed in the config, not the state.
    # firebase_token: Optional[str] #<-- REMOVE THIS LINE


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_gemini)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm",
            self.exists_action,
            {"action": "action", END: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.runnable = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        # This function does not need any changes
        print("\n=== exists_action ===")
        result = state["messages"][-1]
        has_tool_calls = hasattr(result, 'tool_calls') and len(result.tool_calls) > 0
        print(f"Has tool calls: {has_tool_calls}")
        if has_tool_calls:
            print(f"Tool calls: {result.tool_calls}")
        decision = "action" if has_tool_calls else END
        print(f"Decision: {decision}")
        return decision
    
    # --- CHANGE 2: Update call_gemini to accept and use the config ---
    def call_gemini(self, state: AgentState, config: RunnableConfig):
        print("\n=== call_gemini ===")
        
        # Get the token from the config object
        firebase_token = config["configurable"]["firebase_token"]
        print(f"🔍 DEBUG: firebase_token from config: '{'...' if firebase_token else 'None'}'")
        
        messages = state['messages']
        if self.system and not any(isinstance(m, SystemMessage) for m in messages):
            messages = [SystemMessage(content=self.system)] + messages
        
        print("\nInvoking model...")
        message = self.model.invoke(messages)
        print(f"Model response: {message.content[:200]}...")
        if hasattr(message, 'tool_calls'):
            print(f"Tool calls: {message.tool_calls}")
            
        # We no longer need to return the token, as it's not part of the state
        return {'messages': [message]}

    # --- CHANGE 3: Update take_action to accept and use the config ---
    def take_action(self, state: AgentState, config: RunnableConfig):
        print("\n=== take_action ===")

        # Get the token from the config object
        firebase_token = config["configurable"]["firebase_token"]
        print(f"🔍 DEBUG: firebase_token from config: '{'...' if firebase_token else 'None'}'")
        
        tool_calls = state['messages'][-1].tool_calls
        results = []

        for t in tool_calls:
            print(f"\nProcessing tool call: {t}")
            tool_args = dict(t['args'])
            
            # Inject the token from the config variable
            tool_args['firebase_token'] = firebase_token
            
            print(f"Invoking tool {t['name']} with args: {tool_args}")
            try:
                result = self.tools[t['name']].invoke(tool_args)
            except Exception as e:
                print(f"❌ Error invoking tool {t['name']}: {e}")
                result = f"Error invoking tool {t['name']}: {str(e)}"

            results.append(ToolMessage(
                tool_call_id=t['id'], 
                name=t['name'], 
                content=str(result)
            ))

        # We no longer need to return the token, as it's not part of the state
        return {'messages': results}

prompt = """
You are an expert travel agent assistant. Your purpose is to help users with various travel-related tasks.
Always check for existing trips first when a user introduces themselves or asks about travel plans.

**//-- TOOLS --//**
1. check_trips(firebase_token): Get all trips for the current user
2. check_flights(tripId, firebase_token): Get flights for a specific trip
3. check_accom(tripId, firebase_token): Get accommodation details
4. check_activities(tripId, firebase_token): Get activity information

**//-- HOW TO INTERACT --//**
1. Start by calling check_trips(firebase_token) when a user asks about trips or travel plans
2. Ask clarifying questions if needed
3. Always pass the firebase_token to tools
4. Present results clearly
5. Be proactive in suggesting next steps

**//-- START OF CONVERSATION --//**
Begin by checking for trips using the provided firebase_token.
"""

import os
import json
import flask
from dotenv import load_dotenv
from flask_cors import CORS
from flask import request, jsonify
from typing import Dict, Any, Optional
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_google_vertexai import ChatVertexAI
# from travelAgent import (
#     check_trips, 
#     check_flights, 
#     check_accom, 
#     check_activities,
#     Agent,
#     prompt
# )

load_dotenv()
app = flask.Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Configure Vertex AI model
try:
    model = ChatVertexAI(
        model_name='gemini-2.0-flash-lite-001',
        project='nc-travel-462414',
        location='us-central1',
        max_output_tokens=2048,
        temperature=0.2
    )
    print("Vertex AI model configured successfully")
except Exception as e:
    print(f"🔴 ERROR: Model configuration failed: {e}")
    model = None


# @app.route('/api/chat', methods=['POST'])
# def chat_with_agent():
#     try:
#         print("\n🔥 === FLASK ROUTE START ===")
        
#         data = request.get_json()
#         if not data or 'messages' not in data:
#             return jsonify({'error': 'Missing "messages" in request'}), 400

#         auth_header = request.headers.get('Authorization')
#         if not auth_header or not auth_header.startswith('Bearer '):
#             return jsonify({'error': 'Missing or invalid Authorization header'}), 401

#         firebase_token = auth_header.split(' ')[1]
        
#         # DEBUG: Print the extracted token
#         print(f"🔍 DEBUG: Extracted firebase_token: '{firebase_token}' (type: {type(firebase_token)}, length: {len(firebase_token) if firebase_token else 'None'})")

#         raw_messages = data['messages']
#         messages = []
#         for msg in raw_messages:
#             role = msg.get("role")
#             content = msg.get("content")
#             if role == "user":
#                 messages.append(HumanMessage(content=content))
#             elif role == "assistant":
#                 messages.append(AIMessage(content=content))
#             elif role == "system":
#                 messages.append(SystemMessage(content=prompt))
#             else:
#                 raise ValueError(f"Unknown message role: {role}")

#         tools = [
#             check_trips,
#             check_flights,
#             check_accom,
#             check_activities,
#         ]

#         print("Tools loaded:")
#         for idx, t in enumerate(tools):
#             print(f"Tool {idx}: {t.name}")
            
#         agent = Agent(model=model, tools=tools, system=prompt)
        
#         # DEBUG: Print the state being passed to the agent
#         initial_state = {"messages": messages, "firebase_token": firebase_token}
#         print(f"🔍 DEBUG: Initial state passed to agent:")
#         print(f"  - messages: {len(messages)} messages")  
#         print(f"  - firebase_token: '{initial_state.get('firebase_token')}' (type: {type(initial_state.get('firebase_token'))})")
        
#         print(f"🔍 DEBUG: About to invoke agent with state keys: {list(initial_state.keys())}")
        
#         result = agent.runnable.invoke(initial_state)
        
#         print(f"🔍 DEBUG: Agent execution completed")
#         print(f"🔍 DEBUG: Result keys: {list(result.keys())}")

#         updated_history = [{"role": m.type, "content": m.content} for m in result["messages"]]

#         return jsonify({
#             "response": result["messages"][-1].content,
#             "messages": updated_history
#         })

#     except Exception as e:
#         print(f"🔴 ERROR in /api/chat: {e}")
#         import traceback
#         traceback.print_exc()
#         return jsonify({"error": f"Internal error: {str(e)}"}), 500

@app.route('/api/chat', methods=['POST'])
def chat_with_agent():
    try:
        print("\n🔥 === FLASK ROUTE START ===")
        
        data = request.get_json()
        if not data or 'messages' not in data:
            return jsonify({'error': 'Missing "messages" in request'}), 400

        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing or invalid Authorization header'}), 401

        firebase_token = auth_header.split(' ')[1]
        
        raw_messages = data['messages']
        messages = []
        for msg in raw_messages:
            role = msg.get("role")
            content = msg.get("content")
            if role == "user":
                messages.append(HumanMessage(content=content))
            elif role == "assistant":
                messages.append(AIMessage(content=content))
            elif role == "system":
                messages.append(SystemMessage(content=prompt))
            else:
                raise ValueError(f"Unknown message role: {role}")

        # Tools and Agent setup remains the same
        tools = [check_trips, check_flights, check_accom, check_activities]
        agent = Agent(model=model, tools=tools, system=prompt)
        
        # --- THIS IS THE PART TO FIX ---
        
        # 1. Define the input dictionary (just the messages)
        agent_input = {"messages": messages}
        
        # 2. Define the configuration dictionary (with the token)
        agent_config = {"configurable": {"firebase_token": firebase_token}}

        print(f"🔍 DEBUG: Invoking agent with config...")

        # 3. Call invoke() with BOTH the input and the config
        result = agent.runnable.invoke(agent_input, config=agent_config)

        # --- END OF FIX ---
        
        print(f"🔍 DEBUG: Agent execution completed")

        updated_history = [{"role": m.type, "content": m.content} for m in result["messages"]]

        return jsonify({
            "response": result["messages"][-1].content,
            "messages": updated_history
        })

    except Exception as e:
        print(f"🔴 ERROR in /api/chat: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"Internal error: {str(e)}"}), 500

    
# import flask
# from flask_cors import CORS
# from flask import request, jsonify

# # This will print to your terminal the moment Flask loads the file.
# print("✅✅✅ LOADING THE NEW, CORRECT app.py FILE! ✅✅✅")

# app = flask.Flask(__name__)
# CORS(app)

# # A simple test route.
# @app.route('/api/chat', methods=['POST'])
# def chat_with_agent():
#     print("🔥🔥🔥 TEST ROUTE WAS HIT SUCCESSFULLY! 🔥🔥🔥")
#     return jsonify({"message": "Success! Your server is running the new code."})

