"""
Comprehensive test suite for the Continuous Prediction Railway DSS System.
Tests train registration, state management, continuous monitoring, and API endpoints.
"""

import asyncio
import time
import requests
import json
from typing import List, Dict, Any
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContinuousPredictionTester:
    """Test suite for the continuous prediction system."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.test_trains: List[Dict] = []
        self.test_results: Dict[str, Any] = {}
        
    def create_test_trains(self) -> List[Dict]:
        """Create a set of test trains with different characteristics."""
        test_trains = [
            {
                "train_id": "12001",
                "train_type": "express",
                "current_section": "SEC001",
                "from_station": "NDLS",
                "to_station": "HWH",
                "scheduled_arrival": "14:30:00",
                "actual_delay": 5.0,
                "block_length_km": 150.0,
                "speed_limit_kmph": 120.0,
                "rake_length_m": 400.0,
                "priority_level": 1,
                "headway_seconds": 300,
                "tsr_active": "N",
                "tsr_speed_kmph": 0.0
            },
            {
                "train_id": "22413",
                "train_type": "passenger",
                "current_section": "SEC002",
                "from_station": "HWH",
                "to_station": "SRC",
                "scheduled_arrival": "15:45:00",
                "actual_delay": 12.0,  # Higher delay
                "block_length_km": 200.0,
                "speed_limit_kmph": 100.0,
                "rake_length_m": 600.0,
                "priority_level": 2,
                "headway_seconds": 240,  # Shorter headway
                "tsr_active": "Y",  # TSR active
                "tsr_speed_kmph": 60.0
            },
            {
                "train_id": "18001",
                "train_type": "freight",
                "current_section": "SEC003",
                "from_station": "SRC",
                "to_station": "PNBE",
                "scheduled_arrival": "16:15:00",
                "actual_delay": 25.0,  # Very high delay
                "block_length_km": 300.0,
                "speed_limit_kmph": 80.0,
                "rake_length_m": 800.0,
                "priority_level": 3,
                "headway_seconds": 600,
                "tsr_active": "N",
                "tsr_speed_kmph": 0.0
            },
            {
                "train_id": "40001",
                "train_type": "suburban",
                "current_section": "SEC004",
                "from_station": "PNBE",
                "to_station": "ARA",
                "scheduled_arrival": "17:00:00",
                "actual_delay": 2.0,  # Low delay
                "block_length_km": 50.0,
                "speed_limit_kmph": 80.0,
                "rake_length_m": 300.0,
                "priority_level": 2,
                "headway_seconds": 180,  # Very short headway
                "tsr_active": "N",
                "tsr_speed_kmph": 0.0
            }
        ]
        
        self.test_trains = test_trains
        return test_trains
    
    def test_health_endpoint(self) -> bool:
        """Test if the API is running and healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                logger.info(f"✅ Health check passed: {health_data['status']}")
                logger.info(f"Models loaded: {health_data['models']}")
                if 'prediction_engine' in health_data:
                    logger.info(f"Prediction engine: {health_data['prediction_engine']}")
                return True
            else:
                logger.error(f"❌ Health check failed: {response.status_code}")
                return False
        except Exception as e:
            logger.error(f"❌ Health check exception: {e}")
            return False
    
    def test_train_registration(self) -> bool:
        """Test registering trains for continuous monitoring."""
        logger.info("Testing train registration...")
        success_count = 0
        
        for train in self.test_trains:
            try:
                response = requests.post(
                    f"{self.base_url}/trains/register",
                    json=train,
                    timeout=10
                )
                
                if response.status_code == 200:
                    result = response.json()
                    logger.info(f"✅ Registered train {train['train_id']}: {result['status']}")
                    success_count += 1
                else:
                    logger.error(f"❌ Failed to register train {train['train_id']}: {response.status_code}")
                    logger.error(f"Response: {response.text}")
            
            except Exception as e:
                logger.error(f"❌ Exception registering train {train['train_id']}: {e}")
        
        success_rate = success_count / len(self.test_trains)
        logger.info(f"Train registration success rate: {success_rate:.1%} ({success_count}/{len(self.test_trains)})")
        
        return success_rate >= 0.8  # 80% success rate required
    
    def test_train_listing(self) -> bool:
        """Test listing all registered trains."""
        try:
            response = requests.get(f"{self.base_url}/trains", timeout=10)
            
            if response.status_code == 200:
                trains_data = response.json()
                trains_list = trains_data['trains']
                total_count = trains_data['total_count']
                
                logger.info(f"✅ Listed {total_count} trains")
                
                # Check if our test trains are in the list
                registered_train_ids = {train['train_id'] for train in trains_list}
                test_train_ids = {train['train_id'] for train in self.test_trains}
                
                missing_trains = test_train_ids - registered_train_ids
                if missing_trains:
                    logger.warning(f"⚠️ Missing trains in listing: {missing_trains}")
                else:
                    logger.info("✅ All test trains found in listing")
                
                return len(missing_trains) == 0
            
            else:
                logger.error(f"❌ Failed to list trains: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception listing trains: {e}")
            return False
    
    def test_train_details(self) -> bool:
        """Test getting detailed information for specific trains."""
        logger.info("Testing train details retrieval...")
        success_count = 0
        
        for train in self.test_trains:
            try:
                train_id = train['train_id']
                response = requests.get(f"{self.base_url}/trains/{train_id}", timeout=10)
                
                if response.status_code == 200:
                    details = response.json()
                    logger.info(f"✅ Retrieved details for train {train_id}")
                    logger.info(f"   State: {details['current_status']['state']}")
                    logger.info(f"   Last conflict prob: {details['current_status']['last_conflict_prob']}")
                    logger.info(f"   Last delay pred: {details['current_status']['last_delay_pred']}")
                    success_count += 1
                else:
                    logger.error(f"❌ Failed to get details for train {train_id}: {response.status_code}")
            
            except Exception as e:
                logger.error(f"❌ Exception getting details for train {train_id}: {e}")
        
        success_rate = success_count / len(self.test_trains)
        logger.info(f"Train details success rate: {success_rate:.1%}")
        
        return success_rate >= 0.8
    
    def test_train_updates(self) -> bool:
        """Test updating train information."""
        logger.info("Testing train updates...")
        
        # Update the first test train
        train_id = self.test_trains[0]['train_id']
        
        update_data = {
            "train_id": train_id,
            "actual_delay": 15.0,  # Increase delay
            "tsr_active": "Y",     # Activate TSR
            "tsr_speed_kmph": 40.0,
            "priority_level": 1
        }
        
        try:
            response = requests.put(
                f"{self.base_url}/trains/{train_id}/update",
                json=update_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Updated train {train_id}")
                logger.info(f"   Updated fields: {result['updated_fields']}")
                
                # Verify the update by getting train details
                time.sleep(2)  # Wait for update to propagate
                details_response = requests.get(f"{self.base_url}/trains/{train_id}")
                if details_response.status_code == 200:
                    details = details_response.json()
                    current_delay = details['train_data']['actual_delay']
                    logger.info(f"   Verified delay update: {current_delay}")
                    return current_delay == 15.0
                
                return True
            else:
                logger.error(f"❌ Failed to update train {train_id}: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception updating train {train_id}: {e}")
            return False
    
    def test_operator_actions(self) -> bool:
        """Test recording operator actions."""
        logger.info("Testing operator actions...")
        
        train_id = self.test_trains[1]['train_id']  # Use second test train
        
        action_data = {
            "train_id": train_id,
            "action": "hold",
            "operator_id": "OP001",
            "reason": "Traffic congestion ahead"
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/trains/{train_id}/action",
                json=action_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                logger.info(f"✅ Recorded action for train {train_id}")
                logger.info(f"   Action: {result['action']} by {result['operator']}")
                return True
            else:
                logger.error(f"❌ Failed to record action for train {train_id}: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception recording action for train {train_id}: {e}")
            return False
    
    def test_critical_situations(self) -> bool:
        """Test getting critical situations."""
        logger.info("Testing critical situations monitoring...")
        
        try:
            response = requests.get(f"{self.base_url}/critical-situations", timeout=10)
            
            if response.status_code == 200:
                critical_data = response.json()
                logger.info(f"✅ Retrieved critical situations")
                logger.info(f"   Total critical: {critical_data['total_critical']}")
                logger.info(f"   Immediate attention needed: {critical_data['immediate_attention_needed']}")
                
                if critical_data['critical_situations']:
                    logger.info("   Critical trains:")
                    for situation in critical_data['critical_situations']:
                        logger.info(f"     - {situation['train_id']}: "
                                  f"conflict={situation['conflict_probability']:.3f}, "
                                  f"delay={situation['delay_prediction']:.1f}min")
                
                return True
            else:
                logger.error(f"❌ Failed to get critical situations: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception getting critical situations: {e}")
            return False
    
    def test_system_stats(self) -> bool:
        """Test system statistics endpoint."""
        logger.info("Testing system statistics...")
        
        try:
            response = requests.get(f"{self.base_url}/system-stats", timeout=10)
            
            if response.status_code == 200:
                stats = response.json()
                logger.info(f"✅ Retrieved system statistics")
                
                engine_stats = stats['engine_stats']
                logger.info(f"   Engine status: {engine_stats['engine_status']}")
                logger.info(f"   Total trains: {engine_stats['total_trains']}")
                logger.info(f"   Total predictions: {engine_stats['total_predictions']}")
                logger.info(f"   Uptime: {engine_stats['uptime_seconds']:.1f}s")
                
                if 'state_manager_stats' in engine_stats:
                    state_stats = engine_stats['state_manager_stats']
                    if 'state_distribution' in state_stats:
                        logger.info(f"   State distribution: {state_stats['state_distribution']}")
                
                return True
            else:
                logger.error(f"❌ Failed to get system stats: {response.status_code}")
                return False
        
        except Exception as e:
            logger.error(f"❌ Exception getting system stats: {e}")
            return False
    
    def wait_for_predictions(self, duration: int = 60):
        """Wait for predictions to run and observe the system."""
        logger.info(f"Waiting {duration} seconds for continuous predictions...")
        
        start_time = time.time()
        check_interval = 10  # Check every 10 seconds
        
        while time.time() - start_time < duration:
            try:
                # Get system stats
                response = requests.get(f"{self.base_url}/system-stats", timeout=5)
                if response.status_code == 200:
                    stats = response.json()
                    engine_stats = stats['engine_stats']
                    
                    logger.info(f"⏳ Predictions so far: {engine_stats['total_predictions']}")
                    logger.info(f"   Predictions/min: {engine_stats.get('predictions_per_minute', 0):.1f}")
                    
                    # Check for critical situations
                    critical_response = requests.get(f"{self.base_url}/critical-situations", timeout=5)
                    if critical_response.status_code == 200:
                        critical_data = critical_response.json()
                        if critical_data['total_critical'] > 0:
                            logger.warning(f"🚨 {critical_data['total_critical']} trains in CRITICAL state")
                
                time.sleep(check_interval)
                
            except Exception as e:
                logger.warning(f"⚠️ Error during monitoring: {e}")
                time.sleep(check_interval)
        
        logger.info("✅ Monitoring period completed")
    
    def cleanup_test_trains(self):
        """Remove test trains after testing."""
        logger.info("Cleaning up test trains...")
        
        for train in self.test_trains:
            try:
                train_id = train['train_id']
                response = requests.delete(f"{self.base_url}/trains/{train_id}", timeout=10)
                
                if response.status_code == 200:
                    logger.info(f"✅ Removed train {train_id}")
                else:
                    logger.warning(f"⚠️ Failed to remove train {train_id}: {response.status_code}")
            
            except Exception as e:
                logger.warning(f"⚠️ Exception removing train {train_id}: {e}")
    
    def run_complete_test_suite(self):
        """Run the complete test suite."""
        logger.info("=" * 80)
        logger.info("🚂 RAILWAY DSS CONTINUOUS PREDICTION SYSTEM - TEST SUITE")
        logger.info("=" * 80)
        
        # Prepare test data
        self.create_test_trains()
        
        test_results = {}
        
        # Run tests in sequence
        tests = [
            ("Health Check", self.test_health_endpoint),
            ("Train Registration", self.test_train_registration),
            ("Train Listing", self.test_train_listing),
            ("Train Details", self.test_train_details),
            ("Train Updates", self.test_train_updates),
            ("Operator Actions", self.test_operator_actions),
            ("Critical Situations", self.test_critical_situations),
            ("System Statistics", self.test_system_stats)
        ]
        
        for test_name, test_function in tests:
            logger.info(f"\n📋 Running test: {test_name}")
            logger.info("-" * 40)
            
            try:
                result = test_function()
                test_results[test_name] = result
                
                if result:
                    logger.info(f"✅ {test_name}: PASSED")
                else:
                    logger.error(f"❌ {test_name}: FAILED")
            
            except Exception as e:
                logger.error(f"💥 {test_name}: EXCEPTION - {e}")
                test_results[test_name] = False
            
            time.sleep(2)  # Brief pause between tests
        
        # Wait for continuous predictions
        logger.info(f"\n🔄 CONTINUOUS PREDICTION MONITORING")
        logger.info("-" * 40)
        self.wait_for_predictions(60)  # Monitor for 1 minute
        
        # Final system check
        logger.info(f"\n🔍 FINAL SYSTEM CHECK")
        logger.info("-" * 40)
        final_check = self.test_system_stats()
        test_results["Final Check"] = final_check
        
        # Cleanup
        logger.info(f"\n🧹 CLEANUP")
        logger.info("-" * 40)
        self.cleanup_test_trains()
        
        # Results summary
        logger.info(f"\n📊 TEST RESULTS SUMMARY")
        logger.info("=" * 80)
        
        passed_tests = sum(test_results.values())
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests
        
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{test_name:.<30} {status}")
        
        logger.info("-" * 40)
        logger.info(f"Overall Success Rate: {success_rate:.1%} ({passed_tests}/{total_tests})")
        
        if success_rate >= 0.8:
            logger.info("🎉 CONTINUOUS PREDICTION SYSTEM: READY FOR PRODUCTION!")
        elif success_rate >= 0.6:
            logger.warning("⚠️ CONTINUOUS PREDICTION SYSTEM: NEEDS ATTENTION")
        else:
            logger.error("❌ CONTINUOUS PREDICTION SYSTEM: NOT READY")
        
        logger.info("=" * 80)
        
        return test_results

def main():
    """Main function to run the test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Railway DSS Continuous Prediction System")
    parser.add_argument("--url", default="http://localhost:8000", 
                       help="Base URL of the API server")
    parser.add_argument("--quick", action="store_true",
                       help="Run quick tests without extended monitoring")
    
    args = parser.parse_args()
    
    tester = ContinuousPredictionTester(base_url=args.url)
    
    if args.quick:
        # Quick test mode - just basic functionality
        logger.info("🏃 Running quick test mode...")
        tester.create_test_trains()
        
        quick_tests = [
            tester.test_health_endpoint,
            tester.test_train_registration,
            tester.test_train_listing,
            tester.test_system_stats
        ]
        
        results = []
        for test in quick_tests:
            results.append(test())
        
        tester.cleanup_test_trains()
        
        success_rate = sum(results) / len(results)
        logger.info(f"Quick test success rate: {success_rate:.1%}")
    
    else:
        # Full test suite
        tester.run_complete_test_suite()

if __name__ == "__main__":
    main()