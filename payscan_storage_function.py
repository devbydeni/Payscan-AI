"""
PayScan AI Agent - Storage Function
Stores processed invoice results to S3
"""

import json
import boto3
import os
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

s3_client = boto3.client('s3')

class PayScanStorage:
    def __init__(self):
        self.results_bucket = os.environ.get('RESULTS_BUCKET', 'payscan-results-prod')
        self.processing_bucket = os.environ.get('PROCESSING_BUCKET', 'payscan-processing-prod')
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for storage operations"""
        try:
            logger.info(f"Starting storage: {json.dumps(event, default=str)}")
            
            job_id = event['job_id']
            textract_data = event['textract_data']
            sagemaker_data = event['sagemaker_data']
            bedrock_data = event['bedrock_data']
            source_info = event['source_info']
            
            # Combine all processing results
            final_result = {
                'job_id': job_id,
                'processing_timestamp': datetime.utcnow().isoformat(),
                'source': {
                    'bucket': source_info.get('bucket'),
                    'key': source_info.get('key')
                },
                'textract_output': textract_data.get('Payload', {}),
                'sagemaker_output': sagemaker_data.get('Payload', {}),
                'bedrock_output': bedrock_data.get('Payload', {}),
                'status': 'completed'
            }
            
            # Extract key data for easy access
            if bedrock_data.get('Payload', {}).get('success'):
                curated_data = bedrock_data['Payload'].get('curated_invoice_data', {})
                final_result['quick_access'] = {
                    'invoice_id': curated_data.get('invoice_id'),
                    'vendor_name': curated_data.get('vendor_name'),
                    'amount': curated_data.get('amount'),
                    'due_date': curated_data.get('due_date'),
                    'risk_level': curated_data.get('overall_risk_level'),
                    'status': curated_data.get('status')
                }
            
            # Store main result
            main_key = f"processed/{job_id}/final_result.json"
            self.store_to_s3(self.results_bucket, main_key, final_result)
            
            # Store individual outputs for reference
            self.store_to_s3(self.results_bucket, f"processed/{job_id}/textract_output.json", textract_data)
            self.store_to_s3(self.results_bucket, f"processed/{job_id}/sagemaker_output.json", sagemaker_data)
            self.store_to_s3(self.results_bucket, f"processed/{job_id}/bedrock_output.json", bedrock_data)
            
            # Create index for quick lookup
            index_key = f"index/{job_id}.json"
            index_data = {
                'job_id': job_id,
                'invoice_id': final_result.get('quick_access', {}).get('invoice_id'),
                'vendor': final_result.get('quick_access', {}).get('vendor_name'),
                'amount': final_result.get('quick_access', {}).get('amount'),
                'storage_location': f"s3://{self.results_bucket}/{main_key}",
                'timestamp': datetime.utcnow().isoformat()
            }
            self.store_to_s3(self.results_bucket, index_key, index_data)
            
            return {
                'success': True,
                'job_id': job_id,
                'storage_location': f"s3://{self.results_bucket}/{main_key}",
                'files_stored': 5,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in storage: {str(e)}")
            return {
                'success': False,
                'job_id': event.get('job_id'),
                'error': str(e),
                'error_type': 'storage_error'
            }
    
    def store_to_s3(self, bucket: str, key: str, data: Dict[str, Any]) -> None:
        """Store data to S3"""
        try:
            s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=json.dumps(data, default=str, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Stored to s3://{bucket}/{key}")
        except Exception as e:
            logger.error(f"Error storing to S3: {str(e)}")
            raise

storage = PayScanStorage()

def lambda_handler(event, context):
    """Lambda entry point"""
    return storage.lambda_handler(event, context)
