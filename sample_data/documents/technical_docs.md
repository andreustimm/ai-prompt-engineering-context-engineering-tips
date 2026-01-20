# TechCorp API Documentation

## Overview

The TechCorp API provides programmatic access to all TechCorp services. This RESTful API uses JSON for request and response bodies and standard HTTP methods.

**Base URL:** `https://api.techcorp.com/v1`

**Authentication:** All API requests require a valid API key passed in the `Authorization` header.

## Authentication

### Getting Your API Key

1. Log in to your TechCorp dashboard
2. Navigate to Settings > API Keys
3. Click "Generate New Key"
4. Copy and securely store your key

### Using Your API Key

Include your API key in the Authorization header:

```
Authorization: Bearer YOUR_API_KEY
```

## Rate Limits

| Plan | Requests/minute | Requests/day |
|------|-----------------|--------------|
| Basic | 60 | 10,000 |
| Professional | 300 | 100,000 |
| Enterprise | 1,000 | Unlimited |

## Endpoints

### Users

#### List Users

```http
GET /users
```

**Parameters:**
- `page` (optional): Page number for pagination (default: 1)
- `limit` (optional): Results per page (default: 20, max: 100)
- `status` (optional): Filter by status (active, inactive, pending)

**Response:**
```json
{
  "data": [
    {
      "id": "usr_123abc",
      "email": "user@example.com",
      "name": "John Doe",
      "status": "active",
      "created_at": "2024-01-15T10:30:00Z"
    }
  ],
  "meta": {
    "page": 1,
    "limit": 20,
    "total": 150
  }
}
```

#### Get User

```http
GET /users/{user_id}
```

**Response:**
```json
{
  "id": "usr_123abc",
  "email": "user@example.com",
  "name": "John Doe",
  "status": "active",
  "role": "admin",
  "created_at": "2024-01-15T10:30:00Z",
  "updated_at": "2024-03-20T14:45:00Z"
}
```

#### Create User

```http
POST /users
```

**Request Body:**
```json
{
  "email": "newuser@example.com",
  "name": "Jane Smith",
  "role": "member"
}
```

#### Update User

```http
PATCH /users/{user_id}
```

**Request Body:**
```json
{
  "name": "Jane Doe",
  "status": "inactive"
}
```

#### Delete User

```http
DELETE /users/{user_id}
```

### Projects

#### List Projects

```http
GET /projects
```

**Parameters:**
- `owner_id` (optional): Filter by owner
- `status` (optional): Filter by status (active, archived, draft)

#### Create Project

```http
POST /projects
```

**Request Body:**
```json
{
  "name": "My Project",
  "description": "Project description",
  "settings": {
    "visibility": "private",
    "notifications": true
  }
}
```

### Analytics

#### Get Analytics Summary

```http
GET /analytics/summary
```

**Parameters:**
- `start_date` (required): Start date (YYYY-MM-DD)
- `end_date` (required): End date (YYYY-MM-DD)
- `metrics` (optional): Comma-separated list of metrics

**Response:**
```json
{
  "period": {
    "start": "2024-01-01",
    "end": "2024-01-31"
  },
  "metrics": {
    "total_users": 1500,
    "active_users": 1200,
    "new_signups": 150,
    "api_calls": 2500000,
    "data_processed_gb": 45.6
  }
}
```

### Webhooks

#### Register Webhook

```http
POST /webhooks
```

**Request Body:**
```json
{
  "url": "https://your-server.com/webhook",
  "events": ["user.created", "user.updated", "project.created"],
  "secret": "your_webhook_secret"
}
```

#### Webhook Events

| Event | Description |
|-------|-------------|
| user.created | New user registered |
| user.updated | User profile updated |
| user.deleted | User account deleted |
| project.created | New project created |
| project.updated | Project settings changed |
| project.deleted | Project deleted |

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "invalid_request",
    "message": "The request was invalid",
    "details": [
      {
        "field": "email",
        "message": "Invalid email format"
      }
    ]
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| invalid_request | 400 | Request validation failed |
| unauthorized | 401 | Invalid or missing API key |
| forbidden | 403 | Insufficient permissions |
| not_found | 404 | Resource not found |
| rate_limited | 429 | Too many requests |
| server_error | 500 | Internal server error |

## SDKs

Official SDKs are available for:

- **Python:** `pip install techcorp`
- **JavaScript/Node.js:** `npm install @techcorp/sdk`
- **Ruby:** `gem install techcorp`
- **Go:** `go get github.com/techcorp/go-sdk`

### Python Example

```python
from techcorp import TechCorpClient

client = TechCorpClient(api_key="YOUR_API_KEY")

# List users
users = client.users.list(status="active")

# Create project
project = client.projects.create(
    name="New Project",
    description="My awesome project"
)

# Get analytics
analytics = client.analytics.summary(
    start_date="2024-01-01",
    end_date="2024-01-31"
)
```

## Support

- **Documentation:** https://docs.techcorp.com
- **API Status:** https://status.techcorp.com
- **Support Email:** api-support@techcorp.com
- **Community Forum:** https://community.techcorp.com
