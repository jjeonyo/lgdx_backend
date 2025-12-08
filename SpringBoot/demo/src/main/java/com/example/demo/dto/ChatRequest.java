package com.example.demo.dto;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class ChatRequest {
    private String userId;    // 사용자 ID
    private String message;   // 질문 내용 (STT 텍스트)
    private String sessionId; // 세션 ID (room_id로 사용, 예: room_user_001, room_user_002)
    private String source;    // 메시지 출처 ('chat' 또는 'live')
}