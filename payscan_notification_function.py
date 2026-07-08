"""
PayScan AI Agent - Notification Function
Sends processing status updates via SNS and email
"""

import json
import boto3
import os
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

sns_client = boto3.client('sns')

class PayScanNotifier:
    def __init__(self):
        self.sns_topic_arn = os.environ.get('SNS_TOPIC_ARN')
    
    def lambda_handler(self, event: Dict[str, Any], context: Any) -> Dict[str, Any]:
        """Main Lambda handler for notifications"""
        try:
            logger.info(f"Sending notification: {json.dumps(event, default=str)}")
            
            job_id = event['job_id']
            status = event['status']
            
            # Build notification message based on status
            if status == 'completed':
                message = self.build_completion_message(event)
            elif status == 'failed':
                message = self.build_failure_message(event)
            else:
                message = self.build_status_message(event)
            
            # Send SNS notification
            if self.sns_topic_arn:
                response = sns_client.publish(
                    TopicArn=self.sns_topic_arn,
                    Subject=f"PayScan AI Agent - {status.upper()} ({job_id})",
                    Message=message
                )
                logger.info(f"Notification sent: {response['MessageId']}")
            else:
                logger.warning("SNS topic not configured")
            
            return {
                'success': True,
                'job_id': job_id,
                'status': status,
                'notification_sent': True,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error sending notification: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'job_id': event.get('job_id')
            }
    
    def build_completion_message(self, event: Dict[str, Any]) -> str:
        """Build completion notification message"""
        job_id = event['job_id']
        summary = event.get('results_summary', {})
        alerts = event.get('alerts', [])
        
        message = f"""
PayScan AI Agent - Processing Complete ✓

Job ID: {job_id}
Timestamp: {datetime.utcnow().isoformat()}
Status: COMPLETED

Invoice Details:
  Invoice ID: {summary.get('invoice_id', 'N/A')}
  Vendor: {summary.get('vendor', 'N/A')}
  Amount: ${summary.get('amount', 'N/A')}
  Due Date: {summary.get('due_date', 'N/A')}
  Risk Level: {summary.get('risk_level', 'N/A')}

Processing Summary:
  Data Quality Score: {summary.get('data_quality_score', 'N/A')}
  Status: {summary.get('status', 'N/A')}
  Alerts Generated: {len(alerts)}

Alert Details:
"""
        
        if alerts:
            for alert in alerts[:3]:  # Top 3 alerts
                message += f"\n  • [{alert.get('severity', 'INFO').upper()}] {alert.get('title', 'Alert')}"
        else:
            message += "\n  • No alerts"
        
        message += "\n\nReview full report in PayScan dashboard.\n"
        return message
    
    def build_failure_message(self, event: Dict[str, Any]) -> str:
        """Build failure notification message"""
        job_id = event['job_id']
        error_type = event.get('error_type', 'Unknown')
        error_details = event.get('error_details', {})
        
        message = f"""
PayScan AI Agent - Processing Failed ✗

Job ID: {job_id}
Timestamp: {datetime.utcnow().isoformat()}
Status: FAILED

Error Information:
  Error Type: {error_type}
  Error Details: {json.dumps(error_details, indent=4)}

Next Steps:
  1. Review error logs in CloudWatch
  2. Check input file format and size
  3. Verify SageMaker endpoint status
  4. Retry processing or contact support

Job ID for reference: {job_id}
"""
        return message
    
    def build_status_message(self, event: Dict[str, Any]) -> str:
        """Build generic status notification message"""
        job_id = event['job_id']
        status = event.get('status', 'Unknown')
        
        message = f"""
PayScan AI Agent - Status Update

Job ID: {job_id}
Timestamp: {datetime.utcnow().isoformat()}
Status: {status.upper()}

Processing workflow update received.
Check PayScan dashboard for more details.
"""
        return message

notifier = PayScanNotifier()

def lambda_handler(event, context):
    """Lambda entry point"""
    return notifier.lambda_handler(event, context)