package com.example.demo.dto;

import lombok.Getter;
import lombok.Setter;

@Getter @Setter
public class ChatRequest {
    private String userId;    // 사용자 ID
    private String message;   // 질문 내용 (STT 텍스트)
}