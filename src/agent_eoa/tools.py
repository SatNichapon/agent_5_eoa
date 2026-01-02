import logging
import random
from langchain_core.tools import tool

# Configure standard logger
logger = logging.getLogger(__name__)

# Mock Database
calendar_db = {
    "2023-10-20": ["10:00", "11:00"], 
    "2023-10-21": [] 
}

@tool
def check_calendar_availability(date: str) -> str:
    """Checks if a specific date is available for an event."""
    logger.info(f"Tool Call: Checking calendar for {date}")
    
    if date in calendar_db:
        busy_slots = calendar_db[date]
        if not busy_slots:
            return f"Date {date} is completely free."
        return f"Date {date} has busy slots at: {', '.join(busy_slots)}."
    return f"Date {date} is completely free."

@tool
def book_venue(date: str, time: str) -> str:
    """Books the venue for a specific date and time."""
    logger.info(f"Tool Call: Booking venue on {date} at {time}")
    
    # Simulated Failure (20% chance) for ReAct testing
    if random.random() < 0.2: 
        logger.warning("Simulated booking failure triggered.")
        return "Error: Venue is unexpectedly under maintenance."
    
    if date in calendar_db and time in calendar_db[date]:
        return f"Error: Slot {time} on {date} is already booked."
    
    if date not in calendar_db:
        calendar_db[date] = []
    calendar_db[date].append(time)
    
    return f"Success: Venue booked for {date} at {time}."

@tool
def send_email_invitation(recipient_list: str, subject: str) -> str:
    """Sends email invitations."""
    logger.info(f"Tool Call: Sending email to {recipient_list}")
    return f"Success: Sent {len(recipient_list.split(','))} invitations."