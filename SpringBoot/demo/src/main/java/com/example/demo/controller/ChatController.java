package com.example.demo.controller;

import com.example.demo.dto.ChatRequest;
import com.example.demo.dto.ChatResponse;
import com.example.demo.service.ChatService;
import lombok.RequiredArgsConstructor;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api/chatbot") // 가게 주소
@RequiredArgsConstructor
@CrossOrigin(origins = "*") // CORS 허용 (프로덕션에서는 특정 도메인만 허용)
public class ChatController {

    private final ChatService chatService;

    // 앱에서 질문을 보내는 곳 (POST 요청)
    @PostMapping("/ask")
    public ChatResponse ask(@RequestBody ChatRequest request) {
        try {
            System.out.println("📩 [Controller] 질문 도착 - userId: " + request.getUserId() + 
                ", sessionId: " + request.getSessionId() + 
                ", source: " + request.getSource() + 
                ", message: " + request.getMessage());
            
            if (request == null || request.getMessage() == null || request.getMessage().trim().isEmpty()) {
                System.err.println("❌ 잘못된 요청: 메시지가 비어있습니다.");
                ChatResponse errorResponse = new ChatResponse();
                errorResponse.setAnswer("메시지를 입력해주세요.");
                errorResponse.setSources(java.util.Collections.emptyList());
                System.out.println("📤 [Controller] 에러 응답 반환: " + errorResponse.getAnswer());
                return errorResponse;
            }
            
            ChatResponse response = chatService.processChat(request);
            System.out.println("📤 [Controller] 응답 반환 완료 - answer 길이: " + 
                (response.getAnswer() != null ? response.getAnswer().length() : 0) + 
                ", sources 개수: " + (response.getSources() != null ? response.getSources().size() : 0));
            System.out.println("📤 [Controller] 응답 내용 (처음 100자): " + 
                (response.getAnswer() != null ? response.getAnswer().substring(0, Math.min(100, response.getAnswer().length())) : "null"));
            return response;
        } catch (Exception e) {
            System.err.println("❌ [Controller] 예외 발생: " + e.getMessage());
            e.printStackTrace();
            
            ChatResponse errorResponse = new ChatResponse();
            errorResponse.setAnswer("서버 오류가 발생했습니다: " + e.getMessage());
            errorResponse.setSources(java.util.Collections.emptyList());
            return errorResponse;
        }
    }

    // 채팅방 삭제 및 새 room 생성
    @PostMapping("/room/delete")
    public java.util.Map<String, Object> deleteRoom(@RequestBody java.util.Map<String, String> request) {
        try {
            String userId = request.get("userId");
            String roomId = request.get("roomId");
            
            System.out.println("🗑️ [Controller] 채팅방 삭제 요청 - userId: " + userId + ", roomId: " + roomId);
            
            if (userId == null || userId.trim().isEmpty()) {
                throw new RuntimeException("userId가 필요합니다.");
            }
            
            // ChatService의 deleteRoomAndCreateNew 메서드 호출
            String newRoomId = chatService.deleteRoomAndCreateNew(userId, roomId);
            
            java.util.Map<String, Object> response = new java.util.HashMap<>();
            response.put("success", true);
            response.put("message", "채팅방이 삭제되었고 새 채팅방이 생성되었습니다.");
            response.put("newRoomId", newRoomId);
            
            System.out.println("✅ [Controller] 채팅방 삭제 및 새 room 생성 완료 - newRoomId: " + newRoomId);
            return response;
            
        } catch (Exception e) {
            System.err.println("❌ [Controller] 채팅방 삭제 실패: " + e.getMessage());
            e.printStackTrace();
            
            java.util.Map<String, Object> errorResponse = new java.util.HashMap<>();
            errorResponse.put("success", false);
            errorResponse.put("message", "채팅방 삭제 실패: " + e.getMessage());
            return errorResponse;
        }
    }
}