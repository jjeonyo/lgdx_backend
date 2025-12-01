package com.example.demo.service;

import com.example.demo.dto.ChatRequest;
import com.example.demo.dto.ChatResponse;
import com.example.demo.dto.PythonRequest;
import com.google.api.core.ApiFuture;
import com.google.cloud.firestore.Firestore;
import com.google.cloud.firestore.WriteResult;
import com.google.firebase.cloud.FirestoreClient;
import lombok.RequiredArgsConstructor;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;

import java.util.HashMap;
import java.util.Map;

@Service
@RequiredArgsConstructor
public class ChatService {

    private final WebClient webClient = WebClient.create("http://localhost:8000");

    public ChatResponse processChat(ChatRequest request) {
        
        // 1. 파이어베이스 DB 가져오기
        Firestore db = FirestoreClient.getFirestore();
        
        // 방 이름은 편의상 "room_사용자ID"로 고정합니다.
        String roomName = "room_" + request.getUserId();

        // 2. [사용자 질문] 저장 (메시지 1)
        saveMessageToFirebase(db, roomName, "user", request.getMessage());

        // 3. 파이썬(AI)에게 질문하기
        PythonRequest pythonReq = new PythonRequest(request.getUserId(), request.getMessage());
        
        ChatResponse aiResponse = webClient.post()
                .uri("/chat")
                .bodyValue(pythonReq)
                .retrieve()
                .bodyToMono(ChatResponse.class)
                .block();

        // 4. [AI 답변] 저장 (메시지 2)
        if (aiResponse != null) {
            saveMessageToFirebase(db, roomName, "ai", aiResponse.getAnswer());
        }

        return aiResponse;
    }

    // 파이어베이스 저장 도우미 함수
    private void saveMessageToFirebase(Firestore db, String roomName, String sender, String text) {
        try {
            Map<String, Object> message = new HashMap<>();
            message.put("sender", sender); // 누가 (user 또는 ai)
            message.put("text", text);     // 내용
            message.put("timestamp", System.currentTimeMillis()); // 시간

            // chat_rooms -> room_xxx -> messages -> 자동생성ID 문서에 저장
            db.collection("chat_rooms")
                    .document(roomName)
                    .collection("messages")
                    .add(message);
            
            System.out.println("🔥 Firebase 저장 완료: [" + sender + "] " + text);
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}