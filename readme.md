# PayScan AI Agent

> **Intelligent Invoice Processing Automation**  
> Transform your manual invoice processing into an automated, AI-powered workflow that extracts, analyzes, and reports on invoice data with 95%+ accuracy.

[![AWS](https://img.shields.io/badge/AWS-Cloud%20Native-orange)](https://aws.amazon.com/)
[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://python.org/)
[![License](https://img.shields.io/badge/License-GPL-green)](LICENSE)

##  What is PayScan AI Agent?

PayScan AI Agent is a fully automated, serverless AI system that processes invoices from upload to final analysis. It combines multiple AWS AI services to create an intelligent processing pipeline that handles everything from OCR extraction to fraud detection.

### Key Features
- Smart OCR**: Extract text and data from PDF and image invoices
- AI Analysis: Advanced pattern recognition and data validation
- Real-time Processing: Process invoices in under 30 seconds
- Fraud Detection: Identify duplicates and suspicious patterns
- Automated Reporting: Generate insights and recommendations
- Cost Efficient: Pay only for what you process

##  Architecture

```
📁 Invoice Upload (S3)
    ↓
🤖 Lambda Orchestrator
    ↓
📋 Amazon Textract (OCR)
    ↓
🧠 SageMaker AI (Mistral-7B LLM)
    ↓
🔍 Bedrock Agent (Analysis & Curation)
    ↓
💾 Storage & Notifications (S3 + SNS)
```


## 📖 How to Use

### Method 1: S3 Upload (Automatic Processing)

1. **Get Your Upload Bucket Name**
   ```bash
   aws cloudformation describe-stacks \
     --stack-name payscan-ai-agent-prod \
     --query 'Stacks[0].Outputs[?OutputKey==`UploadBucketName`].OutputValue' \
     --output text
   ```

2. **Upload an Invoice**
   ```bash
   aws s3 cp your-invoice.pdf s3://your-upload-bucket-name/
   ```

3. **Processing Happens Automatically**
   - The system detects the upload
   - Starts the processing pipeline
   - Sends email notification when complete

### Method 2: API Gateway (Programmatic Access)

1. **Get Your API URL**
   ```bash
   aws cloudformation describe-stacks \
     --stack-name payscan-ai-agent-prod \
     --query 'Stacks[0].Outputs[?OutputKey==`APIGatewayURL`].OutputValue' \
     --output text
   ```

2. **Upload via API**
   ```bash
   curl -X POST "YOUR_API_URL/upload" \
     -H "Content-Type: application/pdf" \
     --data-binary @your-invoice.pdf
   ```

3. **Check Processing Status**
   ```bash
   curl -X GET "YOUR_API_URL/status/JOB_ID"
   ```

### Method 3: Batch Processing

1. **Upload Multiple Files**
   ```bash
   aws s3 cp invoices/ s3://your-upload-bucket-name/ --recursive
   ```

2. **Monitor Progress**
   ```bash
   aws stepfunctions list-executions \
     --state-machine-arn YOUR_STATE_MACHINE_ARN
   ```

## 📊 Understanding the Results

### Processing Output Structure
```json
{
  "job_id": "unique-job-identifier",
  "invoice_data": {
    "invoice_id": "INV-2024-001",
    "vendor_name": "Acme Corporation",
    "amount": 1250.00,
    "due_date": "2024-02-15",
    "confidence_score": 0.95
  },
  "analysis": {
    "risk_level": "low",
    "duplicates_found": [],
    "alerts": [],
    "recommendations": []
  },
  "processing_stats": {
    "total_time": "24.5s",
    "accuracy_score": 0.97
  }
}
```

### Result Locations
- **Raw Results**: `s3://your-results-bucket/processed/JOB_ID/`
- **Reports**: `s3://your-results-bucket/reports/JOB_ID/`
- **Analytics**: Available through the web dashboard

## ⚙️ Configuration

### Confidence Thresholds
Adjust AI confidence requirements:
```bash
aws lambda update-function-configuration \
  --function-name payscan-sagemaker-prod \
  --environment Variables='{
    "CONFIDENCE_THRESHOLD":"0.85",
    "ENABLE_DUPLICATE_DETECTION":"true"
  }'
```

### Processing Options
- **Auto-processing**: Enable/disable automatic processing on upload
- **Notification settings**: Configure email alerts and thresholds
- **Retention policies**: Set how long to keep processed data

### Cost Controls
```bash
# Set monthly budget alerts
aws budgets create-budget \
  --account-id YOUR_ACCOUNT_ID \
  --budget file://budget-config.json
```

## 🔍 Monitoring & Troubleshooting

### Check System Health
```bash
# Check all Lambda functions
aws lambda list-functions --query 'Functions[?starts_with(FunctionName,`payscan`)].FunctionName'

# Check Step Functions executions
aws stepfunctions list-executions \
  --state-machine-arn YOUR_STATE_MACHINE_ARN \
  --status-filter FAILED
```

### View Logs
```bash
# Lambda function logs
aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/payscan"

# Step Functions execution logs
aws stepfunctions describe-execution --execution-arn EXECUTION_ARN
```

### Common Issues

**Issue**: SageMaker endpoint timeout
```bash
# Solution: Check endpoint status
aws sagemaker describe-endpoint --endpoint-name payscan-mistral-endpoint

# Restart if needed
aws lambda invoke --function-name payscan-orchestrator-prod \
  --payload '{"action": "start_endpoint"}' response.json
```

**Issue**: High processing costs
```bash
# Solution: Enable auto-stop for SageMaker endpoint
aws lambda update-function-configuration \
  --function-name payscan-orchestrator-prod \
  --environment Variables='{"AUTO_STOP_ENDPOINT":"true"}'
```

**Issue**: Processing errors
```bash
# Check CloudWatch logs for detailed error messages
aws logs filter-log-events \
  --log-group-name "/aws/lambda/payscan-orchestrator-prod" \
  --filter-pattern "ERROR"
```



### Feature Requests
1. Describe the use case
2. Explain expected behavior
3. Consider implementation complexity

## 📚 Additional Resources

### Documentation
- [AWS Textract Documentation](https://docs.aws.amazon.com/textract/)
- [Amazon SageMaker Documentation](https://docs.aws.amazon.com/sagemaker/)
- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)

## 🙏 Acknowledgments

- AWS for providing robust AI services
- Mistral AI for the open-source language model
- The open-source community for inspiration and feedback

---


*Transform your invoice processing today - from manual entry to intelligent automation in minutes, not months.*
