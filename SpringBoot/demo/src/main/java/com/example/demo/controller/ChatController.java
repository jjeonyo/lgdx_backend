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
        try {
            System.out.println("📩 [Controller] 질문 도착 - userId: " + request.getUserId() + ", message: " + request.getMessage());
            
            if (request == null || request.getMessage() == null || request.getMessage().trim().isEmpty()) {
                System.err.println("❌ 잘못된 요청: 메시지가 비어있습니다.");
                ChatResponse errorResponse = new ChatResponse();
                errorResponse.setAnswer("메시지를 입력해주세요.");
                errorResponse.setSources(java.util.Collections.emptyList());
                return errorResponse;
            }
            
            return chatService.processChat(request);
        } catch (Exception e) {
            System.err.println("❌ [Controller] 예외 발생: " + e.getMessage());
            e.printStackTrace();
            
            ChatResponse errorResponse = new ChatResponse();
            errorResponse.setAnswer("서버 오류가 발생했습니다: " + e.getMessage());
            errorResponse.setSources(java.util.Collections.emptyList());
            return errorResponse;
        }
    }
}