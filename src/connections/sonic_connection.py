import logging
import os
import json
import requests
import time
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv, set_key
from web3 import Web3
from web3.middleware import geth_poa_middleware
from src.constants.abi import ERC20_ABI
from src.connections.base_connection import BaseConnection, Action, ActionParameter
from src.constants.networks import SONIC_NETWORKS

logger = logging.getLogger("connections.sonic_connection")

# Contract addresses
BERRY_TEMP_AGENT_ADDRESS = "0xF28eC6250Fc5101D814dd78F9b1673b5e3a55cFa"
BERRY_MANAGER_ADDRESS = "0x56516C11f350EeCe25AeA9e36ECd36CB6c71030d"

class SonicConnectionError(Exception):
    """Base exception for Sonic connection errors"""
    pass

class SonicConnection(BaseConnection):
    
    def __init__(self, config: Dict[str, Any]):
        logger.info("Initializing Sonic connection...")
        self._web3 = None
        
        # Get network configuration
        network = config.get("network", "testnet")
        if network not in SONIC_NETWORKS:
            raise ValueError(f"Invalid network '{network}'. Must be one of: {', '.join(SONIC_NETWORKS.keys())}")
            
        network_config = SONIC_NETWORKS[network]
        self.explorer = network_config["scanner_url"]
        self.rpc_url = network_config["rpc_url"]
        
        super().__init__(config)
        self._initialize_web3()
        self.ERC20_ABI = ERC20_ABI
        self.NATIVE_TOKEN = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
        
        # Load contract ABIs
        self._load_contract_abis()
        
        # Initialize contract instances
        self._initialize_contracts()
        
        # Set up account
        try:
            load_dotenv()
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            if private_key:
                self.account = self._web3.eth.account.from_key(private_key)
        except Exception as e:
            logger.warning(f"Could not load account: {e}")
            self.account = None

    def _get_explorer_link(self, tx_hash: str) -> str:
        """Generate block explorer link for transaction"""
        return f"{self.explorer}/tx/{tx_hash}"

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        if not self._web3:
            self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            if not self._web3.is_connected():
                raise SonicConnectionError("Failed to connect to Sonic network")
            
            try:
                chain_id = self._web3.eth.chain_id
                logger.info(f"Connected to network with chain ID: {chain_id}")
            except Exception as e:
                logger.warning(f"Could not get chain ID: {e}")
    
    def _load_contract_abis(self):
        """Load contract ABIs from artifact files"""
        try:
            # Get the project root directory
            project_root = os.getcwd()
            
            # Load BerryTempAgent ABI
            berry_temp_agent_path = os.path.join(project_root, "artifacts", "contracts", "core", "BerryTempAgent.json")
            with open(berry_temp_agent_path, 'r') as f:
                berry_temp_agent_json = json.load(f)
                self.BERRY_TEMP_AGENT_ABI = berry_temp_agent_json["abi"]
            
            # Load BerryManager ABI
            berry_manager_path = os.path.join(project_root, "artifacts", "contracts", "core", "BerryManager.json")
            with open(berry_manager_path, 'r') as f:
                berry_manager_json = json.load(f)
                self.BERRY_MANAGER_ABI = berry_manager_json["abi"]
                
            logger.info("Successfully loaded contract ABIs")
            
        except Exception as e:
            logger.error(f"Failed to load contract ABIs: {e}")
            # Fallback to empty ABIs
            self.BERRY_TEMP_AGENT_ABI = []
            self.BERRY_MANAGER_ABI = []
    
    def _initialize_contracts(self):
        """Initialize contract instances"""
        try:
            # Initialize BerryTempAgent contract
            self.berry_temp_agent = self._web3.eth.contract(
                address=Web3.to_checksum_address(BERRY_TEMP_AGENT_ADDRESS),
                abi=self.BERRY_TEMP_AGENT_ABI
            )
            
            # Initialize BerryManager contract
            self.berry_manager = self._web3.eth.contract(
                address=Web3.to_checksum_address(BERRY_MANAGER_ADDRESS),
                abi=self.BERRY_MANAGER_ABI
            )
            
            logger.info("Successfully initialized contract instances")
            
        except Exception as e:
            logger.error(f"Failed to initialize contract instances: {e}")
            self.berry_temp_agent = None
            self.berry_manager = None

    @property
    def is_llm_provider(self) -> bool:
        return False

    def validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate Sonic configuration from JSON"""
        required = ["network"]
        missing = [field for field in required if field not in config]
        if missing:
            raise ValueError(f"Missing config fields: {', '.join(missing)}")
        
        if config["network"] not in SONIC_NETWORKS:
            raise ValueError(f"Invalid network '{config['network']}'. Must be one of: {', '.join(SONIC_NETWORKS.keys())}")
            
        return config

    def register_actions(self) -> None:
        """Register available actions for the agent"""
        self.actions = {
            # Berry Temperature Monitoring Actions
            "monitor-berry-temperature": Action(
                name="monitor-berry-temperature",
                parameters=[
                    ActionParameter("batch_id", False, int, "Batch ID to monitor"),
                    ActionParameter("temperature", False, float, "Temperature reading"),
                    ActionParameter("location", False, str, "Current location")
                ],
                description="Monitor and analyze temperature data for berry shipments"
            ),
            "manage-berry-quality": Action(
                name="manage-berry-quality",
                parameters=[
                    ActionParameter("batch_id", False, int, "Batch ID to assess")
                ],
                description="Assess and predict berry quality based on temperature history"
            ),
            "process-agent-recommendations": Action(
                name="process-agent-recommendations",
                parameters=[
                    ActionParameter("batch_id", True, int, "Batch ID to process"),
                ],
                description="Process agent recommendations and update supplier reputation"
            ),
            "manage-batch-lifecycle": Action(
                name="manage-batch-lifecycle",
                parameters=[
                    ActionParameter("action", True, str, "Action to perform (create, complete, status)"),
                    ActionParameter("batch_id", False, int, "Batch ID for existing batches"),
                    ActionParameter("berry_type", False, str, "Berry type for new batches")
                ],
                description="Manage berry batch lifecycle from creation to delivery"
            ),
            
            # Original Sonic Actions
            "get-token-by-ticker": Action(
                name="get-token-by-ticker",
                parameters=[
                    ActionParameter("ticker", True, str, "Token ticker symbol to look up")
                ],
                description="Get token address by ticker symbol"
            ),
            "get-balance": Action(
                name="get-balance",
                parameters=[
                    ActionParameter("address", False, str, "Address to check balance for"),
                    ActionParameter("token_address", False, str, "Optional token address")
                ],
                description="Get $S or token balance"
            ),
            "transfer": Action(
                name="transfer",
                parameters=[
                    ActionParameter("to_address", True, str, "Recipient address"),
                    ActionParameter("amount", True, float, "Amount to transfer"),
                    ActionParameter("token_address", False, str, "Optional token address")
                ],
                description="Send $S or tokens"
            ),
            "swap": Action(
                name="swap",
                parameters=[
                    ActionParameter("token_in", True, str, "Input token address"),
                    ActionParameter("token_out", True, str, "Output token address"),
                    ActionParameter("amount", True, float, "Amount to swap"),
                    ActionParameter("slippage", False, float, "Max slippage percentage")
                ],
                description="Swap tokens"
            )
        }

    def configure(self) -> bool:
        logger.info("\n🔷 SONIC CHAIN SETUP")
        if self.is_configured():
            logger.info("Sonic connection is already configured")
            response = input("Do you want to reconfigure? (y/n): ")
            if response.lower() != 'y':
                return True

        try:
            if not os.path.exists('.env'):
                with open('.env', 'w') as f:
                    f.write('')

            private_key = input("\nEnter your wallet private key: ")
            if not private_key.startswith('0x'):
                private_key = '0x' + private_key
            set_key('.env', 'SONIC_PRIVATE_KEY', private_key)

            if not self._web3.is_connected():
                raise SonicConnectionError("Failed to connect to Sonic network")

            account = self._web3.eth.account.from_key(private_key)
            self.account = account
            logger.info(f"\n✅ Successfully connected with address: {account.address}")
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        try:
            load_dotenv()
            if not os.getenv('SONIC_PRIVATE_KEY'):
                if verbose:
                    logger.error("Missing SONIC_PRIVATE_KEY in .env")
                return False

            if not self._web3.is_connected():
                if verbose:
                    logger.error("Not connected to Sonic network")
                return False
            return True

        except Exception as e:
            if verbose:
                logger.error(f"Configuration check failed: {e}")
            return False

    # NEW METHODS TO SUPPORT BERRY ACTIONS
    def send_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a transaction to the blockchain based on transaction data"""
        try:
            load_dotenv()
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            if not private_key:
                raise SonicConnectionError("Missing SONIC_PRIVATE_KEY in .env")
                
            account = self._web3.s.account.from_key(private_key)
            self.account = account  # Make account accessible for other methods
            
            # Extract parameters from tx_data
            contract_address = tx_data.get("contract_address")
            method = tx_data.get("method")
            args = tx_data.get("args", [])
            gas_limit = tx_data.get("gas_limit", 200000)
            
            # Get contract instance
            if contract_address == BERRY_TEMP_AGENT_ADDRESS:
                contract = self.berry_temp_agent
            elif contract_address == BERRY_MANAGER_ADDRESS:
                contract = self.berry_manager
            else:
                contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(contract_address),
                    abi=self.ERC20_ABI  # Default to ERC20_ABI, might need to adjust
                )
            
            # Get contract function
            contract_function = getattr(contract.functions, method)
            
            # Build transaction
            tx = contract_function(*args).build_transaction({
                'from': account.address,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gas': gas_limit,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": tx_hash.hex(),
                "transaction_url": self._get_explorer_link(tx_hash.hex())
            }
                
        except Exception as e:
            logger.error(f"Failed to send transaction: {e}")
            raise

    def call_contract(self, tx_data: Dict[str, Any]) -> Any:
        """Call a contract method (read-only) based on transaction data"""
        try:
            # Extract parameters from tx_data
            contract_address = tx_data.get("contract_address")
            method = tx_data.get("method")
            args = tx_data.get("args", [])
            
            # Get contract instance
            if contract_address == BERRY_TEMP_AGENT_ADDRESS:
                contract = self.berry_temp_agent
            elif contract_address == BERRY_MANAGER_ADDRESS:
                contract = self.berry_manager
            else:
                contract = self._web3.eth.contract(
                    address=Web3.to_checksum_address(contract_address),
                    abi=self.ERC20_ABI  # Default to ERC20_ABI
                )
            
            # Get contract function
            contract_function = getattr(contract.functions, method)
            
            # Call the function
            result = contract_function(*args).call()
            return result
                
        except Exception as e:
            logger.error(f"Failed to call contract: {e}")
            raise

    # Berry contract interaction methods
    def create_batch(self, berry_type: str) -> Dict[str, Any]:
        """Create a new berry batch"""
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
                
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)
            
            # Build transaction
            tx = self.berry_temp_agent.functions.createBatch(berry_type).build_transaction({
                'from': account.address,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gas': 200000,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
            
            # Get batch count to determine batch ID
            batch_count = self.berry_temp_agent.functions.batchCount().call()
            batch_id = batch_count - 1
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": tx_hash.hex(),
                "transaction_url": self._get_explorer_link(tx_hash.hex()),
                "batch_id": batch_id,
                "berry_type": berry_type
            }
            
        except Exception as e:
            logger.error(f"Failed to create batch: {e}")
            raise

    def record_temperature(self, batch_id: int, temperature: int, location: str) -> Dict[str, Any]:
        """Record temperature for a berry batch"""
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
                
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)
            
            # Build transaction
            tx = self.berry_temp_agent.functions.recordTemperature(
                batch_id, temperature, location
            ).build_transaction({
                'from': account.address,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gas': 200000,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": tx_hash.hex(),
                "transaction_url": self._get_explorer_link(tx_hash.hex()),
                "batch_id": batch_id,
                "temperature": temperature,
                "location": location
            }
            
        except Exception as e:
            logger.error(f"Failed to record temperature: {e}")
            raise

    def get_batch_details(self, batch_id: int) -> Dict[str, Any]:
        """Get details for a berry batch"""
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
                
            batch = self.berry_temp_agent.functions.getBatchDetails(batch_id).call()
            
            # Format batch data
            batch_data = {
                "batchId": batch[0],
                "berryType": batch[1],
                "startTime": batch[2],
                "endTime": batch[3],
                "isActive": batch[4],
                "status": batch[5],
                "qualityScore": batch[6],
                "predictedShelfLife": batch[7]
            }
            
            return batch_data
            
        except Exception as e:
            logger.error(f"Failed to get batch details: {e}")
            raise

    def get_temperature_history(self, batch_id: int) -> List[Dict[str, Any]]:
        """Get temperature history for a berry batch"""
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
                
            history = self.berry_temp_agent.functions.getTemperatureHistory(batch_id).call()
            
            # Format temperature readings
            formatted_history = []
            for reading in history:
                formatted_history.append({
                    "timestamp": reading[0],
                    "temperature": reading[1],
                    "location": reading[2],
                    "isBreached": reading[3],
                    "predictedImpact": reading[4]
                })
                
            return formatted_history
            
        except Exception as e:
            logger.error(f"Failed to get temperature history: {e}")
            raise

    def get_agent_predictions(self, batch_id: int) -> List[Dict[str, Any]]:
        """Get agent predictions for a berry batch"""
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
                
            predictions = self.berry_temp_agent.functions.getAgentPredictions(batch_id).call()
            
            # Format predictions
            formatted_predictions = []
            for prediction in predictions:
                formatted_predictions.append({
                    "timestamp": prediction[0],
                    "predictedQuality": prediction[1],
                    "recommendedAction": prediction[2],
                    "actionDescription": prediction[3]
                })
                
            return formatted_predictions
            
        except Exception as e:
            logger.error(f"Failed to get agent predictions: {e}")
            raise

    def register_supplier(self) -> Dict[str, Any]:
        """Register as a supplier"""
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
                
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)
            
            # Build transaction
            tx = self.berry_manager.functions.registerSupplier().build_transaction({
                'from': account.address,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gas': 200000,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": tx_hash.hex(),
                "transaction_url": self._get_explorer_link(tx_hash.hex()),
                "supplier": account.address
            }
            
        except Exception as e:
            logger.error(f"Failed to register supplier: {e}")
            raise

    def process_agent_recommendation(self, batch_id: int) -> Dict[str, Any]:
        """Process agent recommendation for a batch"""
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
                
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)
            
            # Build transaction
            tx = self.berry_manager.functions.processAgentRecommendation(batch_id).build_transaction({
                'from': account.address,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gas': 300000,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": tx_hash.hex(),
                "transaction_url": self._get_explorer_link(tx_hash.hex()),
                "batch_id": batch_id
            }
            
        except Exception as e:
            logger.error(f"Failed to process agent recommendation: {e}")
            raise

    def get_supplier_details(self, supplier_address: Optional[str] = None) -> Dict[str, Any]:
        """Get supplier details"""
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
                
            if not supplier_address:
                private_key = os.getenv('SONIC_PRIVATE_KEY')
                account = self._web3.eth.account.from_key(private_key)
                supplier_address = account.address
                
            details = self.berry_manager.functions.getSupplierDetails(supplier_address).call()
            
            # Format supplier data
            supplier_data = {
                "account": details[0],
                "isRegistered": details[1],
                "reputation": details[2],
                "totalBatches": details[3],
                "successfulBatches": details[4],
                "lastActionTime": details[5]
            }
            
            return supplier_data
            
        except Exception as e:
            logger.error(f"Failed to get supplier details: {e}")
            raise

    def complete_shipment(self, batch_id: int) -> Dict[str, Any]:
        """Complete a shipment"""
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
                
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            account = self._web3.eth.account.from_key(private_key)
            
            # Build transaction
            tx = self.berry_manager.functions.completeShipment(batch_id).build_transaction({
                'from': account.address,
                'nonce': self._web3.eth.get_transaction_count(account.address),
                'gas': 300000,
                'gasPrice': self._web3.eth.gas_price,
                'chainId': self._web3.eth.chain_id
            })
            
            # Sign and send transaction
            signed_tx = account.sign_transaction(tx)
            tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
            
            # Wait for transaction to be mined
            receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash)
            
            return {
                "success": receipt.status == 1,
                "transaction_hash": tx_hash.hex(),
                "transaction_url": self._get_explorer_link(tx_hash.hex()),
                "batch_id": batch_id
            }
            
        except Exception as e:
            logger.error(f"Failed to complete shipment: {e}")
            raise
    
    # Action handler methods
    def monitor_berry_temperature(self, **kwargs) -> Dict[str, Any]:
        """Monitor and analyze temperature data for berry shipments"""
        batch_id = kwargs.get("batch_id", 0)
        temperature = kwargs.get("temperature", 2.5)
        location = kwargs.get("location", "Unknown")
        
        # Convert temperature to integer with 1 decimal precision (* 10)
        temp_int = int(temperature * 10)
        
        return self.record_temperature(batch_id, temp_int, location)
    
    def manage_berry_quality(self, **kwargs) -> Dict[str, Any]:
        """Assess and predict berry quality based on temperature history"""
        batch_id = kwargs.get("batch_id", 0)
        
        # Get batch details
        batch_details = self.get_batch_details(batch_id)
        
        # Get temperature history
        temp_history = self.get_temperature_history(batch_id)
        
        # Get agent predictions
        predictions = self.get_agent_predictions(batch_id)
        
        return {
            "batch_details": batch_details,
            "temperature_history": temp_history,
            "predictions": predictions
        }
    
    def process_agent_recommendations(self, **kwargs) -> Dict[str, Any]:
        """Process agent recommendations and update supplier reputation"""
        batch_id = kwargs.get("batch_id")
        
        return self.process_agent_recommendation(batch_id)
    
    def manage_batch_lifecycle(self, **kwargs) -> Dict[str, Any]:
        """Manage berry batch lifecycle from creation to delivery"""
        action = kwargs.get("action")
        batch_id = kwargs.get("batch_id", 0)
        berry_type = kwargs.get("berry_type", "Strawberry")
        
        if action == "create":
            return self.create_batch(berry_type)
        elif action == "complete":
            return self.complete_shipment(batch_id)
        elif action == "status":
            return self.get_batch_details(batch_id)
        else:
            raise ValueError(f"Unknown action: {action}")

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Sonic action with validation"""
        if action_name not in self.actions:
            raise KeyError(f"Unknown action: {action_name}")

        load_dotenv()
        
        if not self.is_configured(verbose=True):
            raise SonicConnectionError("Sonic is not properly configured")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            raise ValueError(f"Invalid parameters: {', '.join(errors)}")

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        return method(**kwargs)