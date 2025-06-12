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
import uuid
import os
import json
import flask
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from flask_cors import CORS
import requests
import uuid
from datetime import datetime, timezone
from flask import request, jsonify

model = ChatVertexAI(
    model_name='gemini-2.0-flash-lite-001',
    project='nc-travel-462414',
    location='us-central1',
    max_output_tokens=2048,
    temperature=0.2
)



load_dotenv()
app = flask.Flask(__name__)
CORS(app)

class FlightCheckInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip to check flights for.")

class AccomCheckInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip to check accommodation for.")

class ActivityCheckInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip to check activities for.")

class CreateTripInput(BaseModel):
    tripName: str = Field(description="The name of the trip, for example 'Summer Holiday' or 'Work Conference'.")
    location: str = Field(description="The primary city or destination for the trip, for example 'Paris, France'.")
    startDate: str = Field(description="The start date of the trip in 'YYYY-MM-DD' format.")
    endDate: str = Field(description="The end date of the trip in 'YYYY-MM-DD' format.")

class AddFlightInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip to which this flight should be added.")
    airline: str = Field(description="The name of the airline, for example 'British Airways'.")
    flightNumber: str = Field(description="The flight number, for example 'BA283'.")
    departureTime: str = Field(description="The departure date and time in 'YYYY-MM-DDTHH:MM' format (ISO 8601).")
    arrivalTime: str = Field(description="The arrival date and time in 'YYYY-MM-DDTHH:MM' format (ISO 8601).")

class AddAccomInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip to which this accommodation should be added.")
    accommodationName: str = Field(description="The name of the hotel or accommodation, for example 'The Grand Hotel'.")
    checkInDate: str = Field(description="The check-in date in 'YYYY-MM-DD' format.")
    checkOutDate: str = Field(description="The check-out date in 'YYYY-MM-DD' format.")
    address: Optional[str] = Field(description="The address of the accommodation.", default=None)

class AddActivityInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip to which this activity should be added.")
    activityName: str = Field(description="The name of the activity, for example 'Museum Visit' or 'Dinner Reservation'.")
    date: str = Field(description="The date of the activity in 'YYYY-MM-DD' format.")
    time: Optional[str] = Field(description="The time of the activity in 'HH:MM' format.", default=None)

class CreateTripInput(BaseModel):
    tripName: str = Field(description="The name for the new trip, for example 'Summer Holiday' or 'Work Conference'.")
    location: str = Field(description="The primary city or destination for the trip, for example 'Paris, France'.")
    startDate: str = Field(description="The start date of the trip in 'YYYY-MM-DD' format.")
    endDate: str = Field(description="The end date of the trip in 'YYYY-MM-DD' format.")

class SearchFlightsInput(BaseModel):
    originLocationCode: str = Field(description="The IATA code for the departure airport, e.g., 'LHR' for London Heathrow.")
    destinationLocationCode: str = Field(description="The IATA code for the arrival airport, e.g., 'CDG' for Paris Charles de Gaulle.")
    departureDate: str = Field(description="The desired departure date in 'YYYY-MM-DD' format.")
    adults: int = Field(description="The number of adult passengers traveling.", default=1)
    returnDate: Optional[str] = Field(description="The return date in 'YYYY-MM-DD' format. Required for a round-trip flight.", default=None)
    nonStop: Optional[bool] = Field(description="Set to true to search for non-stop flights only.", default=False)

class SearchHotelsInput(BaseModel):
    cityCode: str = Field(description="The IATA city code for where the user wants to find hotels, e.g., 'PAR' for Paris.")
    radius: int = Field(description="The radius around the city center to search for hotels, in kilometers.", default=5)
    checkInDate: Optional[str] = Field(description="The check-in date in 'YYYY-MM-DD' format.", default=None)
    checkOutDate: Optional[str] = Field(description="The check-out date in 'YYYY-MM-DD' format.", default=None)

class SearchActivitiesInput(BaseModel):
    latitude: float = Field(description="The latitude for the center of the search area. e.g., 48.8566 for Paris.")
    longitude: float = Field(description="The longitude for the center of the search area. e.g., 2.3522 for Paris.")
    radius: int = Field(description="The radius around the coordinates to search for activities, in kilometers.", default=1)

class GetItineraryInput(BaseModel):
    tripId: str = Field(description="The unique ID of the trip for which to get the daily itinerary.")




@tool(args_schema=FlightCheckInput)
def check_flights(tripId: str, firebase_token: str):
    """Checks for flight details for a specific trip by its ID."""
    print(f"--- Calling Tool: check_flights for tripId: {tripId} ---")
    try:
        headers = { "Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}" }
        response = requests.get(
            f"http://localhost:3001/api/trips/{tripId}/flights",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve flight information. The server said: {str(e)}"

@tool(args_schema=AccomCheckInput)
def check_accom(tripId: str, firebase_token: str):
    """Checks for accommodation details for a specific trip by its ID."""
    print(f"--- Calling Tool: check_accom for tripId: {tripId} ---")
    # ... (your argument validation is fine) ...
    try:
        headers = { "Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}" }
        response = requests.get(
            f"http://localhost:3001/api/trips/{tripId}/accommodations",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        data = response.json()

        if 'data' in data and isinstance(data['data'], list):
            for activity in data['data']:
                if 'description' in activity and isinstance(activity['description'], str):
                    activity['description'] = activity['description'][:300] + '...'

        return json.dumps(data)
    except requests.exceptions.RequestException as e:
        return f"Error: Could not search for activities. The server said: {str(e)}"

@tool(args_schema=ActivityCheckInput)
def check_activities(tripId: str, firebase_token: str):
    """Checks for activity details for a specific trip by its ID."""
    print(f"--- Calling Tool: check_activities for tripId: {tripId} ---")
    try:
        headers = { "Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}" }
        # ✅ FIX: Added '/trips/' to the URL
        response = requests.get(
            f"http://localhost:3001/api/trips/{tripId}/activities",
            headers=headers,
            timeout=10
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return f"Error: Could not retrieve activity information. The server said: {str(e)}"

@tool(args_schema=CreateTripInput)
def create_trip(tripName: str, location: str, startDate: str, endDate: str, firebase_token: str) -> str:
    """Creates a new trip for the user."""
    print(f"--- Calling Tool: create_trip for {tripName} ---")
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}"}
        payload = {
            "tripName": tripName,
            "location": location,
            "startDate": startDate,
            "endDate": endDate,
        }
        response = requests.post("http://localhost:3001/api/trips", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        trip_id = response.json().get('id', 'Unknown ID')
        return f"Successfully created the trip '{tripName}' to {location}. The new trip ID is {trip_id}."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not create the trip. The server said: {e}"

@tool(args_schema=AddFlightInput)
def add_flight_to_trip(tripId: str, airline: str, flightNumber: str, departureTime: str, arrivalTime: str, firebase_token: str) -> str:
    """Adds a flight to a specific trip using its ID."""
    print(f"--- Calling Tool: add_flight_to_trip for tripId: {tripId} ---")
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}"}
        payload = {
            "airline": airline,
            "flightNumber": flightNumber,
            "departureTime": departureTime,
            "arrivalTime": arrivalTime,
        }
        response = requests.post(f"http://localhost:3001/api/trips/{tripId}/flights", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        flight_id = response.json().get('id', 'Unknown ID')
        return f"Successfully added flight {flightNumber} to trip {tripId}. The new flight ID is {flight_id}."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not add the flight. The server said: {e}"

@tool(args_schema=AddAccomInput)
def add_accommodation_to_trip(tripId: str, accommodationName: str, checkInDate: str, checkOutDate: str, address: str, firebase_token: str) -> str:
    """Adds accommodation to a specific trip using its ID."""
    print(f"--- Calling Tool: add_accommodation_to_trip for tripId: {tripId} ---")
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}"}
        payload = {
            "accommodationName": accommodationName,
            "checkInDate": checkInDate,
            "checkOutDate": checkOutDate,
            "address": address,
        }
        response = requests.post(f"http://localhost:3001/api/trips/{tripId}/accommodations", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        accom_id = response.json().get('id', 'Unknown ID')
        return f"Successfully added accommodation '{accommodationName}' to trip {tripId}. The new accommodation ID is {accom_id}."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not add the accommodation. The server said: {e}"

@tool(args_schema=AddActivityInput)
def add_activity_to_trip(tripId: str, activityName: str, date: str, time: str, firebase_token: str) -> str:
    """Adds an activity to a specific trip using its ID."""
    print(f"--- Calling Tool: add_activity_to_trip for tripId: {tripId} ---")
    try:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {firebase_token}"}
        payload = {
            "activityName": activityName,
            "date": date,
            "time": time,
        }
        response = requests.post(f"http://localhost:3001/api/trips/{tripId}/activities", headers=headers, json=payload, timeout=10)
        response.raise_for_status()
        activity_id = response.json().get('id', 'Unknown ID')
        return f"Successfully added activity '{activityName}' to trip {tripId}. The new activity ID is {activity_id}."
    except requests.exceptions.RequestException as e:
        return f"Error: Could not add the activity. The server said: {e}"

@tool(args_schema=CreateTripInput)
def create_trip(tripName: str, location: str, startDate: str, endDate: str, firebase_token: str) -> str:
    """
    Creates a new trip for the user with the specified name, location, and dates.
    Use this tool when a user explicitly asks to create or book a new trip.
    """
    print(f"--- Calling Tool: create_trip for '{tripName}' ---")
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {firebase_token}"
        }

        payload = {
            "tripName": tripName,
            "location": location,
            "startDate": startDate,
            "endDate": endDate,
        }

        response = requests.post(
            "http://localhost:3001/api/trips", 
            headers=headers, 
            json=payload, 
            timeout=10
        )
        
        response.raise_for_status()
        
        trip_id = response.json().get('id', 'Unknown ID')
        return f"Success! The trip to {location} named '{tripName}' has been created. The new trip ID is {trip_id}."

    except requests.exceptions.RequestException as e:
        print(f"Error creating trip: {e}")
        return f"Error: Could not create the trip. The server responded with an error: {str(e)}"
    except Exception as e:
        # Handle other potential errors
        print(f"An unexpected error occurred in create_trip: {e}")
        return f"An unexpected error occurred while creating the trip: {str(e)}"

@tool(args_schema=SearchFlightsInput)
def search_flights(originLocationCode: str, destinationLocationCode: str, departureDate: str, firebase_token: str, adults: int = 1, returnDate: str = None, nonStop: bool = False) -> str:
    """Searches for flight offers using the Amadeus API via the backend."""
    print(f"--- Calling Tool: search_flights from {originLocationCode} to {destinationLocationCode} ---")
    try:
        headers = {"Authorization": f"Bearer {firebase_token}"}
        params = {
            "originLocationCode": originLocationCode,
            "destinationLocationCode": destinationLocationCode,
            "departureDate": departureDate,
            "adults": adults,
            # ✅ FIX: Convert the boolean to its lowercase string representation
            "nonStop": str(nonStop).lower(),
            "max": 5,
        }
        if returnDate:
            params["returnDate"] = returnDate
            
        response = requests.get("http://localhost:3001/api/amadeus/flights", headers=headers, params=params, timeout=20)
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"Error: Could not search for flights. The server said: {str(e)}"

@tool(args_schema=SearchHotelsInput)
def search_hotels_by_city(cityCode: str, firebase_token: str, radius: int = 5, checkInDate: str = None, checkOutDate: str = None) -> str:
    """Searches for hotels in a city using the Amadeus API via the backend."""
    print(f"--- Calling Tool: search_hotels_by_city for {cityCode} ---")
    try:
        headers = {"Authorization": f"Bearer {firebase_token}"}
        params = {
            "cityCode": cityCode, 
            "radius": radius,
            "pageSize": 5,
        }
        if checkInDate:
            params["checkInDate"] = checkInDate
        if checkOutDate:
            params["checkOutDate"] = checkOutDate

        response = requests.get("http://localhost:3001/api/amadeus/hotels", headers=headers, params=params, timeout=20)
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"Error: Could not search for hotels. The server said: {str(e)}"

@tool(args_schema=SearchActivitiesInput)
def search_activities_by_location(latitude: float, longitude: float, firebase_token: str, radius: int = 1) -> str:
    """Searches for tours and activities near a specific location using the Amadeus API via the backend."""
    print(f"--- Calling Tool: search_activities_by_location ---")
    try:
        headers = {"Authorization": f"Bearer {firebase_token}"}
        params = {
            "latitude": latitude, 
            "longitude": longitude, 
            "radius": radius,
            "pageSize": 5,
        }
        response = requests.get("http://localhost:3001/api/amadeus/activities", headers=headers, params=params, timeout=20)
        response.raise_for_status()
        return json.dumps(response.json())
    except requests.exceptions.RequestException as e:
        return f"Error: Could not search for activities. The server said: {str(e)}"

@tool(args_schema=GetItineraryInput)
def get_trip_itinerary(tripId: str, firebase_token: str) -> str:
    """
    Retrieves a structured, day-by-day itinerary for a specific trip, 
    including any associated flights, accommodations, and activities.
    """
    print(f"--- Calling Tool: get_trip_itinerary for tripId: {tripId} ---")
    
    try:
        headers = {
            "Authorization": f"Bearer {firebase_token}"
        }
        
        response = requests.get(
            f"http://localhost:3001/api/{tripId}/itinerary", 
            headers=headers,
            timeout=15
        )
        
        response.raise_for_status()
        
        return json.dumps(response.json())

    except requests.exceptions.RequestException as e:
        print(f"Error getting itinerary for trip {tripId}: {e}")
        return f"Error: Could not retrieve the itinerary. The server said: {str(e)}"
    except Exception as e:
        print(f"An unexpected error occurred in get_trip_itinerary: {e}")
        return f"An unexpected error occurred while fetching the itinerary: {str(e)}"


    

def travel_agent_prompt(trips: list = None) -> str:
    """
    Generates a system prompt for the travel agent chatbot, 
    correctly formatting Firestore timestamps.
    """
    current_date_str = datetime.now(timezone.utc).strftime("%A, %B %d, %Y")

    # The new, more explicit instructions are added here
    persona_and_rules = f"""
You are a world-class, friendly, and helpful travel agent assistant. Your name is Omni.

**Primary Directive:** Your main goal is to assist users with their travel plans using the tools and information available to you. Your tone should be enthusiastic and proactive.

**Initial Message Rule:**
Your first message to the user must include a welcoming greeting asking how you can help. Do not mention any trip details in your first message unlces they specifically ask you. Example: "Hello! Im your travel assistant. How can I help with your plans today?"

{'-'*20} 
**TOOL USAGE RULES - VERY IMPORTANT:**
1.  When you need to use a tool like `check_flights`, you MUST find the correct `tripId` from the "User's Current Trip Information" section below.
2.  Look for the trip the user is asking about (e.g., if they say "my trip to Rome", find the line for Rome).
3.  From that line, extract the value labeled 'ID'.
4.  Use that value for the `tripId` argument in your tool call.
5.  **DO NOT ask the user for the `tripId`. You can always find it in the information provided.**
{'-'*20}

**How to Use Context:**
After your initial greeting, use the information provided below to answer the user's questions. Today's date is {current_date_str}.
"""

    if trips:
        trips.sort(key=lambda x: x.get('startDate', {}).get('seconds', float('inf')))
        trip_details_list = []
        for trip in trips:
            try:
                start_date = datetime.fromtimestamp(trip['startDate']['seconds'], tz=timezone.utc).strftime('%d %b %Y')
                end_date = datetime.fromtimestamp(trip['endDate']['seconds'], tz=timezone.utc).strftime('%d %b %Y')
                detail = f"- Trip to {trip.get('location', 'N/A')} with ID '{trip.get('id', 'N/A')}' (named '{trip.get('tripName', 'No Name')}') from {start_date} to {end_date}."
                trip_details_list.append(detail)
            except (KeyError, TypeError):
                trip_details_list.append(f"- Trip to {trip.get('location', 'N/A')} with ID '{trip.get('id', 'N/A')}' (date information is incomplete).")
        
        trip_details = "\n".join(trip_details_list)
        context_block = f"--- User's Current Trip Information ---\n{trip_details}\n---"
    else:
        context_block = "--- User's Current Trip Information ---\nNo trip information was provided for this user.\n---"

    return f"{persona_and_rules.strip()}\n{context_block.strip()}"


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    def __init__(self, model: ChatVertexAI, tools: list):
        self.model = model.bind_tools(tools)
        self.tools = {t.name: t for t in tools}
        
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

    def exists_action(self, state: AgentState) -> str:
        """Checks if the AI's last message contains a tool call."""
        print("\n--- Decision ---")
        last_message = state['messages'][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            print("✅ Decision: The agent wants to use a tool.")
            return "action"
        print("✅ Decision: The agent will respond directly.")
        return END

    def call_gemini(self, state: AgentState, config: RunnableConfig) -> dict:
        """The 'brain' of the agent. Generates prompts and calls the AI model."""
        print("\n--- Calling Gemini ---")

        trip_data = config.get("configurable", {}).get("trip_data", [])
        
        system_prompt = travel_agent_prompt(trips=trip_data)
        
        messages = [SystemMessage(content=system_prompt)] + state['messages']

        print("✅ System prompt created with user's trip data.")
        print("Invoking model...")
        response = self.model.invoke(messages)
        print(f"✅ Model responded. Content: '{response.content[:100]}...'")
        
        return {'messages': [response]}

    def take_action(self, state: AgentState, config: RunnableConfig) -> dict:
        print("\n--- Taking Action ---")
        
        firebase_token = config["configurable"]["firebase_token"]
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"🛠️ Executing tool '{t['name']}' with args: {t['args']}")
            tool_args = dict(t['args'])
            tool_args['firebase_token'] = firebase_token
            
            try:
                result = self.tools[t['name']].func(**tool_args)
                # --- END OF FIX ---
                
            except Exception as e:
                print(f"🔴 Error executing tool {t['name']}: {e}")
                result = f"Error: {e}"
            
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        
        return {'messages': results}



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

@app.route('/api/agent', methods=['POST'])
def chat_with_agent():
    try:
        print("\n🔥 === FLASK ROUTE START ===")

        data = request.get_json()
        
        trip_data = data.get('trips', [])

        if not data or 'messages' not in data:
            return jsonify({'error': 'Missing "messages" in request'}), 400

        print(f"🔍 DEBUG: Received data from client: {data}")

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

        tools = [
            check_flights, 
            check_accom, 
            check_activities, 
            add_accommodation_to_trip, 
            add_activity_to_trip, 
            add_flight_to_trip, 
            create_trip, 
            search_activities_by_location, 
            search_flights, 
            search_hotels_by_city, 
            get_trip_itinerary, 
            TavilySearchResults
        ]
        
        agent = Agent(model=model, tools=tools)
        
        agent_input = {"messages": messages}

        agent_config = {
            "configurable": {
                "firebase_token": firebase_token,
                "trip_data": trip_data
            }
        }

        print(f"🔍 DEBUG: Invoking agent with config...")

        result = agent.runnable.invoke(agent_input, config=agent_config)
        
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


from datetime import datetime

def travel_chatbot_prompt(trips: list = None) -> str:
    """
    Generates a system prompt for the travel agent chatbot.

    Args:
        trips (list, optional): A list of trip dictionaries. 
                                Each dict should have details like 
                                'destination', 'start_date', 'end_date'. 
                                Defaults to None.

    Returns:
        str: The fully formatted system prompt.
    """
    current_date = datetime.now().strftime("%A, %B %d, %Y")

    persona_and_rules = f"""
You are a world-class, friendly, and helpful travel agent assistant.

**Primary Directive:** Your main goal is to assist users with their travel plans, answer questions about their trips, and help them book new ones.

**Initial Message Rule:**
Your first message to the user, and ONLY your first message, must be a brief, welcoming greeting asking how you can help. Do not mention any trip details in your first message.
Good examples:
- "Hello! How can I assist with your travel plans today?"
- "Welcome! I'm your AI travel assistant. What can I do for you?"

**How to Use Context:**
After your initial greeting, use the information provided below as context to answer the user's questions. Refer to the current date to understand how far away the trips are.
Current Date: {current_date}
"""
    if trips:
        trip_details = "\n".join(
            [
                f"- Trip to {trip.get('location', 'N/A')}: from {trip.get('startDate', 'N/A')} to {trip.get('endDate', 'N/A')}."
                for trip in trips
            ]
        )
        context_block = f"""
--- User's Current Trip Information ---
The user has the following trips booked. Use this information to answer their questions, but do not bring it up unless they ask.
{trip_details}
---
"""
    else:
        context_block = f"""
--- User's Current Trip Information ---
No trip information was provided for this user. If the user asks about their trips, politely inform them that you don't have any on file and ask if you can help them book a new adventure.
---
"""
    system_prompt = f"{persona_and_rules.strip()}\n{context_block.strip()}"
    print(system_prompt)
    return system_prompt

@app.route('/api/chat', methods=['POST'])
def chat():
    
    if not model:
        return jsonify({"error": "Model not configured"}), 500

    data = request.get_json()
    if not data or 'messages' not in data:
        return jsonify({"error": "Request body must be a JSON object with a 'messages' key."}), 400
    
    trips = data['trips']
    
    SYSTEM_PROMPT = travel_chatbot_prompt(trips)

    print(SYSTEM_PROMPT)
    
    raw_messages = data['messages']
    messages = [SystemMessage(content=SYSTEM_PROMPT)]

    for msg in raw_messages:
        if msg.get("role") == "user":
            messages.append(HumanMessage(content=msg.get("content")))
        elif msg.get("role") == "assistant":
            messages.append(AIMessage(content=msg.get("content")))

    try:
        # Get a direct response from the model
        response = model.invoke(messages)
        
        # Send the AI's text content back to the frontend
        return jsonify({"response": response.content})

    except Exception as e:
        print(f"🔴 ERROR invoking model: {e}")
        return jsonify({"error": "Failed to get a response from the AI model."}), 500

if __name__ == '__main__':
    app.run(port=8002, debug=True)