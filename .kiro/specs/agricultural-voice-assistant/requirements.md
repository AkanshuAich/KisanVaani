# Requirements Document

## Introduction

KisanVaani is a voice-first agricultural assistant system that enables farmers to access government subsidies, crop guidance, and loan information through simple voice interactions. Farmers can call a number or use a lightweight mobile app to ask questions in their local language. The system processes the query using a RAG-based AI pipeline and responds with accurate voice output. Basic proactive alerts for new schemes and deadlines are also supported.

## Glossary

- **Voice_Interface**: Handles speech-to-text and text-to-speech conversion
- **RAG_Pipeline**: Retrieves documents from the Knowledge_Base and generates contextual answers
- **Knowledge_Base**: Stores government schemes, crop guidance, and official documents
- **Notification_Service**: Sends outbound voice alerts to farmers
- **Query_Handler**: Coordinates query processing and information retrieval
- **Farmer_Profile**: Stores phone number, location, and crop type for basic personalization

## Requirements

### Requirement 1: Voice Input Processing

**User Story:** As a farmer, I want to speak my question in my local language, so that I can access information easily.

#### Acceptance Criteria

1. WHEN a farmer speaks into the phone or mobile app, THE Voice_Interface SHALL convert the speech to text within 5 seconds
2. WHEN speech is unclear or incomplete, THE Voice_Interface SHALL prompt the farmer to repeat their query
3. THE Voice_Interface SHALL support at least 1 to 2 regional languages for MVP

### Requirement 2: Information Retrieval (Core RAG)

**User Story:** As a farmer, I want accurate information about subsidies and crop schemes, so that I can make informed decisions.

#### Acceptance Criteria

1. WHEN a farmer query is received, THE Query_Handler SHALL process it through the RAG_Pipeline
2. WHEN the query is processed, THE RAG_Pipeline SHALL retrieve relevant documents from the Knowledge_Base
3. WHEN documents are retrieved, THE RAG_Pipeline SHALL generate a response using only the retrieved documents
4. WHEN the Knowledge_Base lacks information for a query, THE System SHALL inform the farmer clearly

### Requirement 3: Voice Output Generation

**User Story:** As a farmer, I want to hear answers in my language, so that I can understand the information without reading.

#### Acceptance Criteria

1. WHEN a response is generated, THE Voice_Interface SHALL convert the text response to speech
2. WHEN converting to speech, THE Voice_Interface SHALL produce clear and simple voice output
3. WHEN a farmer requests, THE Voice_Interface SHALL repeat the last response

### Requirement 4: Basic Proactive Notifications

**User Story:** As a farmer, I want alerts about important schemes and deadlines, so that I don't miss opportunities.

#### Acceptance Criteria

1. WHEN a new scheme is added to the Knowledge_Base, THE Notification_Service SHALL identify eligible farmers based on their Farmer_Profile
2. WHEN eligible farmers are identified, THE Notification_Service SHALL initiate outbound voice calls to notify them
3. WHEN a farmer does not answer, THE Notification_Service SHALL retry the call once

### Requirement 5: Knowledge Base Management

**User Story:** As an administrator, I want to upload official documents for accurate responses, so that farmers receive reliable information.

#### Acceptance Criteria

1. THE Knowledge_Base SHALL store official PDFs and structured scheme data
2. WHEN new documents are uploaded, THE System SHALL process them into embeddings for retrieval
3. WHEN documents are updated, THE System SHALL allow administrators to replace existing documents

### Requirement 6: Farmer Identification (Simplified)

**User Story:** As a farmer, I want personalized responses without complex login steps, so that I can access information quickly.

#### Acceptance Criteria

1. WHEN a farmer calls or uses the app, THE System SHALL identify them using their phone number
2. WHEN a farmer is identified, THE System SHALL retrieve their Farmer_Profile containing location and crop type
3. THE System SHALL store basic profile data without requiring biometric authentication

### Requirement 7: Basic Conversation Context

**User Story:** As a farmer, I want to ask simple follow-up questions, so that I don't have to repeat information.

#### Acceptance Criteria

1. WHEN a farmer asks follow-up questions, THE Query_Handler SHALL maintain conversation context for 3 to 5 turns
2. WHEN context is ambiguous, THE Query_Handler SHALL ask clarifying questions before retrieving information

### Requirement 8: Accessibility

**User Story:** As a farmer with limited literacy or technical skills, I want simple and intuitive interaction, so that I can use the system without assistance.

#### Acceptance Criteria

1. THE Voice_Interface SHALL provide clear voice prompts at each step
2. THE System SHALL support phone-based access via a toll-free number
3. WHEN using the mobile app, THE System SHALL provide a minimal interface with large touch-friendly buttons

### Requirement 9: Performance

**User Story:** As a farmer, I want the system to respond quickly and reliably, so that I can get information when needed.

#### Acceptance Criteria

1. THE System SHALL support up to 1,000 concurrent farmers in initial deployment
2. THE System SHALL maintain basic logging for monitoring system health
3. WHEN system errors occur, THE System SHALL provide a fallback message to the farmer

## Out of Scope for MVP (Future Phase)

The following features are explicitly excluded from the MVP and will be considered for future phases:

- MVP limited to 1-2 languages
- Deep government API integrations for real-time data synchronization
- Market price and weather API integration
- Large-scale concurrency support (10,000+ concurrent users)
- Advanced compliance automation and data privacy features
- Voice biometrics and advanced authentication
- Multi-retry notification logic with complex scheduling
- Detailed analytics and reporting dashboards

## Notes

- The MVP focuses on core voice interaction and RAG-based information retrieval
- System prioritizes simplicity and accessibility for farmers with limited technical skills
- Phone-based access ensures reach to farmers without smartphones
- Basic notification support helps farmers stay informed about new schemes
