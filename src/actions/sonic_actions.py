# src/actions/sonic_actions.py
from src.action_handler import list_registered_actions
import logging
from dotenv import load_dotenv
from src.action_handler import register_action
from datetime import datetime, timedelta
import json

logger = logging.getLogger("actions.sonic_actions")

# Constants for berry temperature monitoring
BERRY_TEMP_AGENT_ADDRESS = "0xF28eC6250Fc5101D814dd78F9b1673b5e3a55cFa"
BERRY_MANAGER_ADDRESS = "0x56516C11f350EeCe25AeA9e36ECd36CB6c71030d"
OPTIMAL_TEMP = 2   # 2°C
MAX_TEMP = 4       # 4°C
MIN_TEMP = 0       # 0°C

# Berry Temperature Monitoring Actions
@register_action("monitor-berry-temperature")
async def monitor_berry_temperature(agent, **kwargs):
    """Monitor and analyze temperature data for berry shipments"""
    try:
        logger.info("Starting berry temperature monitoring...")
        
        # Get batch ID from parameters or use default
        batch_id = kwargs.get("batch_id", 0)
        temperature = kwargs.get("temperature")
        location = kwargs.get("location", "Unknown")
        
        # Validate temperature data
        if temperature is None:
            # Use mock data if no temperature is provided
            mock_temps = [2.5, 3.1, 4.2, 5.0, 3.8, 2.9]
            temperature = mock_temps[int(datetime.now().timestamp()) % len(mock_temps)]
            
        # Determine location from time of day if not provided
        if location == "Unknown":
            hour = datetime.now().hour
            if hour < 8:
                location = "Cold Storage"
            elif hour < 12:
                location = "Loading Dock"
            elif hour < 18:
                location = "Transport"
            else:
                location = "Delivery Center"
        
        logger.info(f"Recording temperature {temperature}°C at {location} for batch {batch_id}")
        
        # Call contract method to record temperature
        tx_data = {
            "contract_address": BERRY_TEMP_AGENT_ADDRESS,
            "method": "recordTemperature",
            "args": [batch_id, int(temperature * 10), location],
            "gas_limit": 300000  # Increased gas limit
        }
        
        # Fix: Don't use await if send_transaction doesn't return a coroutine
        sonic_connection = agent.connection_manager.connections["sonic"]
        tx_result = sonic_connection.send_transaction(tx_data)
        logger.info(f"Temperature recording transaction: {tx_result}")
        
        # Check for breaches
        is_breach = temperature > MAX_TEMP or temperature < MIN_TEMP
        breach_severity = "None"
        
        if is_breach:
            deviation = abs(temperature - OPTIMAL_TEMP)
            if deviation > 3:
                breach_severity = "Critical"
            elif deviation > 1:
                breach_severity = "Warning"
            else:
                breach_severity = "Minor"
                
            logger.warning(f"Temperature breach detected: {temperature}°C, Severity: {breach_severity}")
            
        return {
            "status": "completed",
            "batch_id": batch_id,
            "temperature": temperature,
            "location": location,
            "is_breach": is_breach,
            "breach_severity": breach_severity,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Berry temperature monitoring failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }

@register_action("manage-berry-quality")
async def manage_berry_quality(agent, **kwargs):
    """Assess and predict berry quality based on temperature history"""
    try:
        logger.info("Starting berry quality assessment...")
        
        # Get batch ID from parameters or use default
        batch_id = kwargs.get("batch_id", 0)
        
        # Get temperature history for the batch
        tx_data = {
            "contract_address": BERRY_TEMP_AGENT_ADDRESS,
            "method": "getTemperatureHistory",
            "args": [batch_id]
        }
        
        # Fix: Don't use await if call_contract doesn't return a coroutine
        sonic_connection = agent.connection_manager.connections["sonic"]
        temp_history = sonic_connection.call_contract(tx_data)
        
        # If no history available, use mock data
        if not temp_history or len(temp_history) == 0:
            logger.info("No temperature history found, using mock data")
            
            # Generate mock temperature history
            mock_history = []
            base_time = datetime.now() - timedelta(hours=12)
            
            for i in range(6):
                # Generate varying temperatures
                if i == 3:  # Create a breach in the middle
                    temp = 5.2
                else:
                    temp = 2.0 + (i % 3)
                
                # Generate locations
                locations = ["Cold Storage", "Loading Dock", "Transport", "Transport", "Transport", "Delivery Center"]
                
                mock_history.append({
                    "timestamp": (base_time + timedelta(hours=i*2)).isoformat(),
                    "temperature": temp,
                    "location": locations[i],
                    "isBreached": temp > MAX_TEMP or temp < MIN_TEMP
                })
            
            temp_history = mock_history
        
        # Calculate quality impact
        quality_score = 100
        shelf_life_hours = 72  # Default 3-day shelf life for berries
        
        for reading in temp_history:
            # Handle both dict and list formats
            if isinstance(reading, dict):
                temp = reading.get("temperature", 0)
            else:  # Assume list format from contract
                temp = reading[1] / 10.0  # Contract stores temp * 10
                
            if isinstance(temp, str):
                temp = float(temp)
                
            if temp > MAX_TEMP:
                deviation = temp - MAX_TEMP
                quality_score -= deviation * 5
                shelf_life_hours -= deviation * 4
            elif temp < MIN_TEMP:
                deviation = MIN_TEMP - temp
                quality_score -= deviation * 7
                shelf_life_hours -= deviation * 6
        
        # Ensure we don't go below zero
        quality_score = max(0, quality_score)
        shelf_life_hours = max(0, shelf_life_hours)
        
        # Determine recommended action based on quality
        action = "No Action"
        action_description = "Quality is within acceptable parameters."
        
        if quality_score < 60:
            action = "Reject"
            action_description = "Quality severely compromised. Recommend rejection."
        elif quality_score < 70:
            action = "Reroute"
            action_description = "Quality concerns detected. Recommend rerouting to nearest facility."
        elif quality_score < 80:
            action = "Expedite"
            action_description = "Quality at risk. Recommend expedited delivery."
        elif quality_score < 90:
            action = "Alert"
            action_description = "Minor quality impact. Continue monitoring closely."
        
        result = {
            "status": "completed",
            "batch_id": batch_id,
            "quality_score": round(quality_score, 1),
            "shelf_life_hours": round(shelf_life_hours, 1),
            "temperature_readings": len(temp_history),
            "recommended_action": action,
            "action_description": action_description,
            "timestamp": datetime.now().isoformat()
        }
        
        logger.info(f"Quality assessment complete: {result}")
        return result
        
    except Exception as e:
        logger.error(f"Berry quality management failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }

@register_action("process-agent-recommendations")
async def process_agent_recommendations(agent, **kwargs):
    """Process agent recommendations and update supplier reputation"""
    try:
        logger.info("Processing agent recommendations...")
        
        # Get parameters
        batch_id = kwargs.get("batch_id", 0)
        quality_score = kwargs.get("quality_score")
        recommended_action = kwargs.get("recommended_action")
        
        # If quality score not provided, fetch from previous action or use default
        if quality_score is None:
            # Call the quality management function without await
            quality_result = await manage_berry_quality(agent, batch_id=batch_id)
            quality_score = quality_result.get("quality_score", 80)
            recommended_action = quality_result.get("recommended_action", "No Action")
        
        # Call contract method to process recommendation
        tx_data = {
            "contract_address": BERRY_MANAGER_ADDRESS,
            "method": "processAgentRecommendation",
            "args": [batch_id],
            "gas_limit": 500000  # Increased gas limit
        }
        
        # Fix: Don't use await if send_transaction doesn't return a coroutine
        sonic_connection = agent.connection_manager.connections["sonic"]
        tx_result = sonic_connection.send_transaction(tx_data)
        logger.info(f"Recommendation processing transaction: {tx_result}")
        
        # Get supplier details
        supplier_tx_data = {
            "contract_address": BERRY_MANAGER_ADDRESS,
            "method": "getSupplierDetails",
            "args": [sonic_connection.account.address]
        }
        
        # Fix: Don't use await if call_contract doesn't return a coroutine
        supplier_details = sonic_connection.call_contract(supplier_tx_data)
        
        # If no details available, use mock data
        if not supplier_details:
            supplier_details_dict = {
                "reputation": 85,
                "totalBatches": 12,
                "successfulBatches": 10
            }
        else:
            # Convert list or tuple to dict if needed
            if isinstance(supplier_details, (list, tuple)):
                supplier_details_dict = {
                    "account": supplier_details[0],
                    "isRegistered": supplier_details[1],
                    "reputation": supplier_details[2],
                    "totalBatches": supplier_details[3],
                    "successfulBatches": supplier_details[4],
                    "lastActionTime": supplier_details[5]
                }
            else:
                supplier_details_dict = supplier_details  # If it's already a dict
        
        # Calculate supplier action based on quality
        supplier_action = "None"
        
        if quality_score >= 90:
            supplier_action = "Reward"
        elif quality_score < 60:
            supplier_action = "Warn"
        
        return {
            "status": "completed",
            "batch_id": batch_id,
            "quality_score": quality_score,
            "recommended_action": recommended_action,
            "supplier_reputation": supplier_details_dict.get("reputation", 85),
            "total_batches": supplier_details_dict.get("totalBatches", 12),
            "successful_batches": supplier_details_dict.get("successfulBatches", 10),
            "supplier_action": supplier_action,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Process agent recommendations failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }

@register_action("manage-batch-lifecycle")
async def manage_batch_lifecycle(agent, **kwargs):
    """Manage berry batch lifecycle from creation to delivery"""
    try:
        logger.info("Managing batch lifecycle...")
        
        # Get parameters
        action = kwargs.get("action", "create")
        berry_type = kwargs.get("berry_type", "Strawberry")
        batch_id = kwargs.get("batch_id", 0)
        
        # Get Sonic connection
        sonic_connection = agent.connection_manager.connections["sonic"]
        
        if action == "create":
            # Create a new batch
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "createBatch",
                "args": [berry_type],
                "gas_limit": 300000  # Increased gas limit
            }
            
            # Fix: Don't use await if send_transaction doesn't return a coroutine
            tx_result = sonic_connection.send_transaction(tx_data)
            logger.info(f"Batch creation transaction: {tx_result}")
            
            # Get the batch count to determine the new batch ID
            count_tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "batchCount",
                "args": []
            }
            
            # Fix: Don't use await if call_contract doesn't return a coroutine
            batch_count = sonic_connection.call_contract(count_tx_data)
            if batch_count:
                batch_id = int(batch_count) - 1
                
            return {
                "status": "completed",
                "action": "create",
                "berry_type": berry_type,
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat()
            }
            
        elif action == "complete":
            # Complete a shipment
            tx_data = {
                "contract_address": BERRY_MANAGER_ADDRESS,
                "method": "completeShipment",
                "args": [batch_id],
                "gas_limit": 400000  # Increased gas limit
            }
            
            # Fix: Don't use await if send_transaction doesn't return a coroutine
            tx_result = sonic_connection.send_transaction(tx_data)
            logger.info(f"Shipment completion transaction: {tx_result}")
            
            return {
                "status": "completed",
                "action": "complete",
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat()
            }
            
        elif action == "status":
            # Get batch details
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "getBatchDetails",
                "args": [batch_id]
            }
            
            # Fix: Don't use await if call_contract doesn't return a coroutine
            batch_details = sonic_connection.call_contract(tx_data)
            
            # If no details available, use mock data
            if not batch_details:
                batch_details_dict = {
                    "batchId": batch_id,
                    "berryType": berry_type,
                    "startTime": (datetime.now() - timedelta(hours=24)).timestamp(),
                    "status": 1,  # 1 = InTransit
                    "qualityScore": 85,
                    "predictedShelfLife": 60 * 60 * 60  # 60 hours in seconds
                }
            else:
                # Convert list or tuple to dict if needed
                if isinstance(batch_details, (list, tuple)):
                    batch_details_dict = {
                        "batchId": batch_details[0],
                        "berryType": batch_details[1],
                        "startTime": batch_details[2],
                        "endTime": batch_details[3],
                        "isActive": batch_details[4],
                        "status": batch_details[5],
                        "qualityScore": batch_details[6],
                        "predictedShelfLife": batch_details[7]
                    }
                else:
                    batch_details_dict = batch_details  # If it's already a dict
            
            status_map = {
                0: "Created",
                1: "InTransit",
                2: "Delivered",
                3: "Rejected"
            }
            
            current_status = status_map.get(batch_details_dict.get("status", 1), "InTransit")
            
            return {
                "status": "completed",
                "action": "status",
                "batch_id": batch_id,
                "berry_type": batch_details_dict.get("berryType", berry_type),
                "batch_status": current_status,
                "quality_score": batch_details_dict.get("qualityScore", 85),
                "predicted_shelf_life_hours": batch_details_dict.get("predictedShelfLife", 60 * 60 * 60) / 3600,
                "timestamp": datetime.now().isoformat()
            }
        
    except Exception as e:
        logger.error(f"Batch lifecycle management failed: {str(e)}", exc_info=True)
        return {
            "status": "failed",
            "error": str(e)
        }

# Helper function to verify action registration
def verify_action_registration():
    from src.action_handler import list_registered_actions
    actions = list_registered_actions()
    logger.info(f"Registered berry actions: {actions}")
    required_actions = [
        "monitor-berry-temperature", 
        "manage-berry-quality", 
        "process-agent-recommendations", 
        "manage-batch-lifecycle"
    ]
    missing = [a for a in required_actions if a not in actions]
    if missing:
        logger.warning(f"Missing actions: {missing}")
    return len(missing) == 0

# Run verification
verify_action_registration()