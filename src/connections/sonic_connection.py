import logging
import os
import json
import time
from datetime import datetime, timedelta
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
        self.account = None
        
        # Enable mock mode if specified
        self.use_mock_mode = config.get("use_mock_mode", False)
        if self.use_mock_mode:
            logger.info("📌 Running in MOCK MODE - No real blockchain transactions will be made")
        
        # Get network configuration
        network = config.get("network", "testnet")
        if network not in SONIC_NETWORKS:
            raise ValueError(f"Invalid network '{network}'. Must be one of: {', '.join(SONIC_NETWORKS.keys())}")
            
        network_config = SONIC_NETWORKS[network]
        self.explorer = network_config["scanner_url"]
        self.rpc_url = network_config["rpc_url"]
        
        super().__init__(config)
        
        # Initialize web3, contracts, and account
        self._initialize_web3()
        self.ERC20_ABI = ERC20_ABI
        self.NATIVE_TOKEN = "0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE"
        self._load_contract_abis()
        self._initialize_contracts()
        self._load_account()

    def _get_explorer_link(self, tx_hash: str) -> str:
        """Generate block explorer link for transaction"""
        return f"{self.explorer}/tx/{tx_hash}"

    def _initialize_web3(self):
        """Initialize Web3 connection"""
        if not self._web3:
            self._web3 = Web3(Web3.HTTPProvider(self.rpc_url))
            self._web3.middleware_onion.inject(geth_poa_middleware, layer=0)
            
            if not self._web3.is_connected() and not self.use_mock_mode:
                raise SonicConnectionError("Failed to connect to Sonic network")
            
            try:
                chain_id = self._web3.eth.chain_id
                logger.info(f"Connected to network with chain ID: {chain_id}")
            except Exception as e:
                if not self.use_mock_mode:
                    logger.warning(f"Could not get chain ID: {e}")
    
    def _load_account(self):
        """Load account from private key"""
        try:
            load_dotenv()
            private_key = os.getenv('SONIC_PRIVATE_KEY')
            if private_key:
                self.account = self._web3.eth.account.from_key(private_key)
                logger.info(f"Account loaded with address: {self.account.address}")
                
                # Log account balance for debugging
                if not self.use_mock_mode:
                    try:
                        balance = self._web3.eth.get_balance(self.account.address)
                        logger.info(f"Account balance: {self._web3.from_wei(balance, 'ether')} Sonic Tokens")
                    except Exception as e:
                        logger.warning(f"Could not get account balance: {e}")
        except Exception as e:
            logger.warning(f"Could not load account: {e}")
    
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
            if not self.use_mock_mode:
                logger.error(f"Failed to initialize contract instances: {e}")
            self.berry_temp_agent = None
            self.berry_manager = None

    def get_optimal_gas_price(self):
        """Get optimal gas price for Sonic Blaze Testnet"""
        # Set a minimum gas price high enough to avoid "underpriced" errors
        # For Sonic Blaze Testnet, 10 gwei seems to be the minimum
        minimum_gas_price = self._web3.to_wei(10, 'gwei')
        
        # Get current network gas price
        network_gas_price = self._web3.eth.gas_price
        
        # Use the higher of the two values
        optimal_price = max(minimum_gas_price, network_gas_price * 2)
        logger.info(f"Using gas price: {self._web3.from_wei(optimal_price, 'gwei')} gwei")
        return optimal_price

    def _estimate_transaction_gas(self, tx_params, contract_func, args):
        """Estimate gas for a transaction and check if enough balance is available"""
        try:
            # Create transaction object for estimation
            tx_for_estimate = contract_func(*args).build_transaction(tx_params)
            
            # Estimate gas with a 20% buffer
            estimated_gas = int(self._web3.eth.estimate_gas(tx_for_estimate) * 1.2)
            
            # Get optimal gas price
            gas_price = self.get_optimal_gas_price()
            
            # Calculate total cost
            total_cost = estimated_gas * gas_price
            
            # Get account balance
            balance = self._web3.eth.get_balance(self.account.address)
            
            # Log for debugging
            logger.info(f"Estimated gas: {estimated_gas}")
            logger.info(f"Gas price: {self._web3.from_wei(gas_price, 'gwei')} gwei")
            logger.info(f"Total cost: {self._web3.from_wei(total_cost, 'ether')} Sonic Tokens")
            logger.info(f"Account balance: {self._web3.from_wei(balance, 'ether')} Sonic Tokens")
            
            return {
                "has_funds": balance >= total_cost,
                "gas_limit": estimated_gas,
                "gas_price": gas_price,
                "total_cost": total_cost,
                "balance": balance
            }
        except Exception as e:
            logger.warning(f"Gas estimation failed: {e}")
            # Return conservative defaults
            return {
                "has_funds": True,  # Assume funds are available if estimation fails
                "gas_limit": 500000,  # Higher default gas limit for safety
                "gas_price": self.get_optimal_gas_price(),
                "total_cost": 0,
                "balance": 0
            }

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

            if not self._web3.is_connected() and not self.use_mock_mode:
                raise SonicConnectionError("Failed to connect to Sonic network")

            account = self._web3.eth.account.from_key(private_key)
            self.account = account
            logger.info(f"\n✅ Successfully connected with address: {account.address}")
            
            # Check balance after setup
            if not self.use_mock_mode:
                try:
                    balance = self._web3.eth.get_balance(account.address)
                    logger.info(f"Account balance: {self._web3.from_wei(balance, 'ether')} Sonic Tokens")
                except Exception as e:
                    logger.warning(f"Could not get account balance: {e}")
            
            return True

        except Exception as e:
            logger.error(f"Configuration failed: {e}")
            return False

    def is_configured(self, verbose: bool = False) -> bool:
        if self.use_mock_mode:
            return True
            
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

    # Transaction method with retry
    def send_transaction(self, tx_data: Dict[str, Any]) -> Dict[str, Any]:
        """Send a transaction to the blockchain with retry mechanism"""
        # Mock mode support
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Simulated transaction for method {tx_data.get('method')}")
            return {
                "success": True,
                "transaction_hash": "0x" + "0" * 64,  # Mock transaction hash
                "transaction_url": "#mock-transaction",
                "mock": True
            }
        
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                load_dotenv()
                if not self.account:
                    private_key = os.getenv('SONIC_PRIVATE_KEY')
                    if not private_key:
                        raise SonicConnectionError("Missing SONIC_PRIVATE_KEY in .env")
                    self.account = self._web3.eth.account.from_key(private_key)
                
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
                        abi=self.ERC20_ABI
                    )
                
                # Get contract function
                contract_function = getattr(contract.functions, method)
                
                # Calculate nonce (increase with each retry to avoid nonce conflicts)
                nonce = self._web3.eth.get_transaction_count(self.account.address) + retry_count
                
                # Create transaction parameters
                tx_params = {
                    'from': self.account.address,
                    'nonce': nonce,
                    'chainId': self._web3.eth.chain_id
                }
                
                # Estimate gas and check funds
                gas_info = self._estimate_transaction_gas(tx_params, contract_function, args)
                
                if not gas_info["has_funds"]:
                    raise SonicConnectionError(
                        f"Insufficient funds for transaction. Have: {self._web3.from_wei(gas_info['balance'], 'ether')} tokens, "
                        f"Need: {self._web3.from_wei(gas_info['total_cost'], 'ether')} tokens"
                    )
                
                # Update transaction parameters with gas settings
                # Increase gas limit with each retry
                gas_limit = int(gas_info["gas_limit"] * (1 + (retry_count * 0.2)))
                # Use optimal gas price
                gas_price = self.get_optimal_gas_price()
                
                tx_params.update({
                    'gas': gas_limit,
                    'gasPrice': gas_price
                })
                
                # Build final transaction
                tx = contract_function(*args).build_transaction(tx_params)
                
                # Log transaction details
                logger.info(f"Sending transaction: method={method}, gas={tx['gas']}, gasPrice={self._web3.from_wei(tx['gasPrice'], 'gwei')} gwei")
                
                # Sign and send transaction
                signed_tx = self.account.sign_transaction(tx)
                tx_hash = self._web3.eth.send_raw_transaction(signed_tx.rawTransaction)
                
                # Wait for transaction to be mined with timeout
                receipt = self._web3.eth.wait_for_transaction_receipt(tx_hash, timeout=120)
                
                return {
                    "success": receipt.status == 1,
                    "transaction_hash": tx_hash.hex(),
                    "transaction_url": self._get_explorer_link(tx_hash.hex()),
                    "gas_used": receipt.gasUsed,
                    "effective_gas_price": receipt.effectiveGasPrice if hasattr(receipt, 'effectiveGasPrice') else tx['gasPrice']
                }
                
            except Exception as e:
                logger.warning(f"Transaction attempt {retry_count + 1} failed: {e}")
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(2 * retry_count)  # Exponential backoff
        
        # All retries failed
        logger.error(f"Failed to send transaction after {max_retries} attempts: {last_error}")
        return {
            "success": False,
            "error": str(last_error),
            "method": tx_data.get("method", "unknown")
        }

    def call_contract(self, tx_data: Dict[str, Any]) -> Any:
        """Call a contract method (read-only) based on transaction data"""
        # Mock mode support
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Simulated contract call for method {tx_data.get('method')}")
            # Return mock data based on the method
            method = tx_data.get("method")
            batch_id = tx_data.get("args", [0])[0] if tx_data.get("args") else 0
            
            if method == "getTemperatureHistory":
                # Mock temperature history
                mock_history = []
                base_time = int(datetime.now().timestamp())
                
                for i in range(5):
                    temp = 2.5
                    if i == 2:  # Create one breach for testing
                        temp = 5.5
                    
                    locations = ["Cold Storage", "Loading Dock", "Transport", "Transport", "Delivery Center"]
                    
                    mock_history.append([
                        base_time - 3600 * i,  # timestamp
                        int(temp * 10),        # temperature (x10 for precision)
                        locations[i % len(locations)],  # location
                        temp > 4.0 or temp < 0.0,  # isBreached
                        5 if temp > 4.0 or temp < 0.0 else 0  # predictedImpact
                    ])
                
                return mock_history
            
            elif method == "getBatchDetails":
                # Mock batch details
                return [
                    batch_id,  # batchId
                    "Strawberry",  # berryType
                    int(datetime.now().timestamp()) - 86400,  # startTime
                    0,  # endTime
                    True,  # isActive
                    1,  # status (1 = InTransit)
                    85,  # qualityScore
                    60 * 60 * 60  # predictedShelfLife (60 hours in seconds)
                ]
                
            elif method == "getAgentPredictions":
                # Mock predictions
                return [
                    [
                        int(datetime.now().timestamp()) - 1800,  # timestamp
                        85,  # predictedQuality
                        3,  # recommendedAction
                        "Monitor closely. Continue with standard delivery procedures."  # actionDescription
                    ]
                ]
                
            elif method == "getSupplierDetails":
                # Mock supplier details
                return [
                    "0x" + "1" * 40,  # account
                    True,  # isRegistered
                    85,  # reputation
                    12,  # totalBatches
                    10,  # successfulBatches
                    int(datetime.now().timestamp()) - 3600  # lastActionTime
                ]
                
            elif method == "batchCount":
                # Mock batch count
                return batch_id + 1
                
            # Default mock return
            return []
            
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
                    abi=self.ERC20_ABI
                )
            
            # Get contract function
            contract_function = getattr(contract.functions, method)
            
            # Call the function
            result = contract_function(*args).call()
            return result
                
        except Exception as e:
            logger.error(f"Failed to call contract: {e}")
            raise

    # Berry contract interaction methods with improved error handling
    def create_batch(self, berry_type: str) -> Dict[str, Any]:
        """Create a new berry batch"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Creating batch for berry type: {berry_type}")
            return {
                "success": True,
                "transaction_hash": "0x" + "0" * 64,
                "transaction_url": "#mock-transaction",
                "batch_id": 1,
                "berry_type": berry_type,
                "mock": True
            }
            
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
            
            # Use send_transaction method with proper gas handling
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "createBatch",
                "args": [berry_type]
            }
            
            result = self.send_transaction(tx_data)
            
            if result.get("success", False):
                try:
                    # Get batch count to determine batch ID
                    batch_count = self.berry_temp_agent.functions.batchCount().call()
                    batch_id = batch_count - 1
                    result["batch_id"] = batch_id
                except Exception as e:
                    logger.warning(f"Could not get batch ID: {e}")
                    result["batch_id"] = 0
                    
                result["berry_type"] = berry_type
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to create batch: {e}")
            return {
                "success": False,
                "error": str(e),
                "berry_type": berry_type
            }

    def record_temperature(self, batch_id: int, temperature: int, location: str) -> Dict[str, Any]:
        """Record temperature for a berry batch"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Recording temperature {temperature/10.0}°C at {location} for batch {batch_id}")
            return {
                "success": True,
                "transaction_hash": "0x" + "0" * 64,
                "transaction_url": "#mock-transaction",
                "batch_id": batch_id,
                "temperature": temperature,
                "location": location,
                "mock": True
            }
            
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
            
            # Use send_transaction method with proper gas handling
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "recordTemperature",
                "args": [batch_id, temperature, location]
            }
            
            result = self.send_transaction(tx_data)
            
            if result.get("success", False):
                result["batch_id"] = batch_id
                result["temperature"] = temperature
                result["location"] = location
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to record temperature: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_id": batch_id,
                "temperature": temperature,
                "location": location
            }

    def get_batch_details(self, batch_id: int) -> Dict[str, Any]:
        """Get details for a berry batch"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Getting details for batch {batch_id}")
            return {
                "batchId": batch_id,
                "berryType": "Strawberry",
                "startTime": int(datetime.now().timestamp()) - 86400,
                "endTime": 0,
                "isActive": True,
                "status": 1,  # 1 = InTransit
                "qualityScore": 85,
                "predictedShelfLife": 60 * 60 * 60  # 60 hours in seconds
            }
            
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
            
            # Use call_contract method
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "getBatchDetails",
                "args": [batch_id]
            }
            
            batch = self.call_contract(tx_data)
            
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
            return {
                "error": str(e),
                "batchId": batch_id,
                "berryType": "Unknown",
                "isActive": False,
                "qualityScore": 0
            }

    def get_temperature_history(self, batch_id: int) -> List[Dict[str, Any]]:
        """Get temperature history for a berry batch"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Getting temperature history for batch {batch_id}")
            # Mock temperature history
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
                    "timestamp": int((base_time + timedelta(hours=i*2)).timestamp()),
                    "temperature": int(temp * 10),
                    "location": locations[i],
                    "isBreached": temp > 4.0 or temp < 0.0,
                    "predictedImpact": 5 if (temp > 4.0 or temp < 0.0) else 0
                })
            
            return mock_history
            
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
            
            # Use call_contract method
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "getTemperatureHistory",
                "args": [batch_id]
            }
            
            history = self.call_contract(tx_data)
            
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
            return []

    def get_agent_predictions(self, batch_id: int) -> List[Dict[str, Any]]:
        """Get agent predictions for a berry batch"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Getting agent predictions for batch {batch_id}")
            return [{
                "timestamp": int(datetime.now().timestamp()) - 1800,
                "predictedQuality": 85,
                "recommendedAction": 3,
                "actionDescription": "Monitor closely. Continue with standard delivery procedures."
            }]
            
        try:
            if not self.berry_temp_agent:
                raise SonicConnectionError("BerryTempAgent contract not initialized")
            
            # Use call_contract method
            tx_data = {
                "contract_address": BERRY_TEMP_AGENT_ADDRESS,
                "method": "getAgentPredictions",
                "args": [batch_id]
            }
            
            predictions = self.call_contract(tx_data)
            
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
            return []

    def register_supplier(self) -> Dict[str, Any]:
        """Register as a supplier"""
        if self.use_mock_mode:
            logger.info("MOCK MODE: Registering supplier")
            return {
                "success": True,
                "transaction_hash": "0x" + "0" * 64,
                "transaction_url": "#mock-transaction",
                "supplier": "0x" + "1" * 40,
                "mock": True
            }
            
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
            
            # Use send_transaction method with proper gas handling
            tx_data = {
                "contract_address": BERRY_MANAGER_ADDRESS,
                "method": "registerSupplier",
                "args": []
            }
            
            result = self.send_transaction(tx_data)
            
            if result.get("success", False) and self.account:
                result["supplier"] = self.account.address
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to register supplier: {e}")
            return {
                "success": False,
                "error": str(e)
            }

    def process_agent_recommendation(self, batch_id: int) -> Dict[str, Any]:
        """Process agent recommendation for a batch"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Processing agent recommendation for batch {batch_id}")
            return {
                "success": True,
                "transaction_hash": "0x" + "0" * 64,
                "transaction_url": "#mock-transaction",
                "batch_id": batch_id,
                "mock": True
            }
            
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
            
            # Check if supplier is registered first
            supplier_details = self.get_supplier_details()
            if not supplier_details.get("isRegistered", False):
                logger.warning("Supplier not registered, attempting to register first")
                register_result = self.register_supplier()
                if not register_result.get("success", False):
                    raise SonicConnectionError("Failed to register supplier before processing recommendation")
            
            # Use send_transaction method with proper gas handling
            tx_data = {
                "contract_address": BERRY_MANAGER_ADDRESS,
                "method": "processAgentRecommendation",
                "args": [batch_id]
            }
            
            result = self.send_transaction(tx_data)
            
            if result.get("success", False):
                result["batch_id"] = batch_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process agent recommendation: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_id": batch_id
            }

    def get_supplier_details(self, supplier_address: Optional[str] = None) -> Dict[str, Any]:
        """Get supplier details"""
        if self.use_mock_mode:
            logger.info("MOCK MODE: Getting supplier details")
            return {
                "account": supplier_address or "0x" + "1" * 40,
                "isRegistered": True,
                "reputation": 85,
                "totalBatches": 12,
                "successfulBatches": 10,
                "lastActionTime": int(datetime.now().timestamp()) - 3600
            }
            
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
            
            if not supplier_address and self.account:
                supplier_address = self.account.address
            elif not supplier_address:
                private_key = os.getenv('SONIC_PRIVATE_KEY')
                account = self._web3.eth.account.from_key(private_key)
                supplier_address = account.address
            
            # Use call_contract method
            tx_data = {
                "contract_address": BERRY_MANAGER_ADDRESS,
                "method": "getSupplierDetails",
                "args": [supplier_address]
            }
            
            details = self.call_contract(tx_data)
            
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
            return {
                "account": supplier_address or "Unknown",
                "isRegistered": False,
                "reputation": 0,
                "error": str(e)
            }

    def complete_shipment(self, batch_id: int) -> Dict[str, Any]:
        """Complete a shipment"""
        if self.use_mock_mode:
            logger.info(f"MOCK MODE: Completing shipment for batch {batch_id}")
            return {
                "success": True,
                "transaction_hash": "0x" + "0" * 64,
                "transaction_url": "#mock-transaction",
                "batch_id": batch_id,
                "mock": True
            }
            
        try:
            if not self.berry_manager:
                raise SonicConnectionError("BerryManager contract not initialized")
            
            # Use send_transaction method with proper gas handling
            tx_data = {
                "contract_address": BERRY_MANAGER_ADDRESS,
                "method": "completeShipment",
                "args": [batch_id]
            }
            
            result = self.send_transaction(tx_data)
            
            if result.get("success", False):
                result["batch_id"] = batch_id
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to complete shipment: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_id": batch_id
            }
    
    # Action handler methods with improved error handling
    def monitor_berry_temperature(self, **kwargs) -> Dict[str, Any]:
        """Monitor and analyze temperature data for berry shipments"""
        try:
            batch_id = kwargs.get("batch_id", 0)
            temperature = kwargs.get("temperature", 2.5)
            location = kwargs.get("location", "Unknown")
            
            # Convert temperature to integer with 1 decimal precision (* 10)
            temp_int = int(temperature * 10)
            
            logger.info(f"Monitoring temperature for batch {batch_id}: {temperature}°C at {location}")
            
            # Check account balance before proceeding (if not in mock mode)
            if not self.use_mock_mode and self.account:
                try:
                    balance = self._web3.eth.get_balance(self.account.address)
                    logger.info(f"Current balance: {self._web3.from_wei(balance, 'ether')} Sonic Tokens")
                    
                    if balance == 0:
                        logger.warning("Account has zero balance! Transaction likely to fail.")
                except Exception as e:
                    logger.warning(f"Could not check balance: {e}")
            
            # Record temperature
            return self.record_temperature(batch_id, temp_int, location)
            
        except Exception as e:
            logger.error(f"Berry temperature monitoring failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_id": kwargs.get("batch_id", 0),
                "temperature": kwargs.get("temperature", 0),
                "location": kwargs.get("location", "")
            }
    
    def manage_berry_quality(self, **kwargs) -> Dict[str, Any]:
        """Assess and predict berry quality based on temperature history"""
        try:
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
        except Exception as e:
            logger.error(f"Berry quality management failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_id": kwargs.get("batch_id", 0)
            }
    
    def process_agent_recommendations(self, **kwargs) -> Dict[str, Any]:
        """Process agent recommendations and update supplier reputation"""
        try:
            batch_id = kwargs.get("batch_id")
            return self.process_agent_recommendation(batch_id)
        except Exception as e:
            logger.error(f"Processing agent recommendations failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "batch_id": kwargs.get("batch_id", 0)
            }
    
    def manage_batch_lifecycle(self, **kwargs) -> Dict[str, Any]:
        """Manage berry batch lifecycle from creation to delivery"""
        try:
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
        except Exception as e:
            logger.error(f"Batch lifecycle management failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "action": kwargs.get("action", ""),
                "batch_id": kwargs.get("batch_id", 0)
            }

    def perform_action(self, action_name: str, kwargs) -> Any:
        """Execute a Sonic action with validation"""
        if action_name not in self.actions:
            return {
                "success": False,
                "error": f"Unknown action: {action_name}"
            }

        load_dotenv()
        
        if not self.is_configured(verbose=True) and not self.use_mock_mode:
            return {
                "success": False,
                "error": "Sonic is not properly configured"
            }

        # Log action details for debugging
        logger.info(f"Executing action '{action_name}' with parameters: {kwargs}")
        
        # Check account balance before proceeding with any action
        if self.account and not self.use_mock_mode:
            try:
                balance = self._web3.eth.get_balance(self.account.address)
                logger.info(f"Current account balance: {self._web3.from_wei(balance, 'ether')} Sonic Tokens")
                
                if balance == 0:
                    logger.warning("Account has zero balance! Transactions likely to fail.")
            except Exception as e:
                logger.warning(f"Could not check balance: {e}")

        action = self.actions[action_name]
        errors = action.validate_params(kwargs)
        if errors:
            return {
                "success": False,
                "error": f"Invalid parameters: {', '.join(errors)}"
            }

        method_name = action_name.replace('-', '_')
        method = getattr(self, method_name)
        
        try:
            result = method(**kwargs)
            logger.info(f"Action '{action_name}' completed successfully")
            return result
        except Exception as e:
            logger.error(f"Action '{action_name}' failed: {e}")
            # Return error information instead of raising
            return {
                "success": False,
                "error": str(e),
                "action": action_name,
                "parameters": kwargs
            }