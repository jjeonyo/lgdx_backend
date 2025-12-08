package com.example.demo.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import lombok.AllArgsConstructor;
import lombok.Getter;

@Getter @AllArgsConstructor
public class PythonRequest {
    // 파이썬은 snake_case를 좋아하므로 이름표를 붙여줍니다.
    @JsonProperty("user_id")
    private String userId;

    @JsonProperty("user_message")
    private String userMessage;
    
    @JsonProperty("session_id")
    private String sessionId; // room_id (예: room_user_001, room_user_002)
}