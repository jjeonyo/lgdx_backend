package com.example.demo.controller;

import com.example.demo.dto.ChatRequest;
import com.example.demo.dto.ChatResponse;
import com.example.demo.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chatbot") // 가게 주소
@RequiredArgsConstructor
public class ChatController {

    private final ChatService chatService;

    // 앱에서 질문을 보내는 곳 (POST 요청)
    @PostMapping("/ask")
    public ChatResponse ask(@RequestBody ChatRequest request) {
        System.out.println("📩 질문 도착: " + request.getMessage());
        return chatService.processChat(request);
    }
}