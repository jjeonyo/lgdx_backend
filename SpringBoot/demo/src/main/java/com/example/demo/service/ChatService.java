package com.example.demo.service;

import com.example.demo.dto.ChatRequest;
import com.example.demo.dto.ChatResponse;
import com.example.demo.dto.PythonRequest;
import com.google.cloud.firestore.FieldValue;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.FirebaseApp;
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
        try {
            System.out.println("🔵 [ChatService] 요청 처리 시작 - userId: " + request.getUserId() + ", message: " + request.getMessage());
            
            // 1. Firebase 초기화 확인
            if (FirebaseApp.getApps().isEmpty()) {
                System.err.println("❌ Firebase가 초기화되지 않았습니다!");
                throw new RuntimeException("Firebase가 초기화되지 않았습니다. FirebaseConfig에서 초기화를 확인하세요.");
            }
            System.out.println("✅ Firebase 초기화 확인됨");
            
            // 2. 파이어베이스 DB 가져오기
            Firestore db = FirestoreClient.getFirestore();
            if (db == null) {
                throw new RuntimeException("Firestore 연결 실패 - FirestoreClient.getFirestore()가 null을 반환했습니다.");
            }
            System.out.println("✅ Firestore 연결 성공");
            
            // 방 이름은 편의상 "room_사용자ID"로 고정합니다.
            String roomName = "room_" + request.getUserId();
            System.out.println("📁 채팅방: " + roomName);

            // 2. [사용자 질문] 저장은 Python 서버에서 처리하므로 여기서는 저장하지 않음
            // (Python 서버의 chat_endpoint에서 save_to_firebase를 호출함)

            // 3. 파이썬(AI)에게 질문하기
            PythonRequest pythonReq = new PythonRequest(request.getUserId(), request.getMessage());
            System.out.println("📤 Python 서버로 요청 전송: http://localhost:8000/chat");
            
            ChatResponse aiResponse;
            try {
                aiResponse = webClient.post()
                        .uri("/chat")
                        .bodyValue(pythonReq)
                        .retrieve()
                        .onStatus(status -> status.is4xxClientError() || status.is5xxServerError(), 
                            clientResponse -> {
                                System.err.println("❌ Python 서버 HTTP 에러: " + clientResponse.statusCode());
                                return clientResponse.bodyToMono(String.class)
                                    .map(body -> {
                                        System.err.println("에러 응답 본문: " + body);
                                        throw new RuntimeException("Python 서버 HTTP " + clientResponse.statusCode() + " 에러: " + body);
                                    });
                            })
                        .bodyToMono(ChatResponse.class)
                        .doOnError(error -> {
                            System.err.println("❌ Python 서버 연결 실패: " + error.getClass().getSimpleName() + " - " + error.getMessage());
                            if (error.getCause() != null) {
                                System.err.println("원인: " + error.getCause().getMessage());
                            }
                            error.printStackTrace();
                        })
                        .block();
            } catch (Exception e) {
                System.err.println("❌ Python 서버 통신 중 예외: " + e.getClass().getSimpleName() + " - " + e.getMessage());
                e.printStackTrace();
                
                // ConnectException이나 TimeoutException이 원인으로 있는지 확인
                Throwable cause = e.getCause();
                if (cause instanceof java.net.ConnectException) {
                    throw new RuntimeException("Python 서버(포트 8000)에 연결할 수 없습니다. 서버가 실행 중인지 확인하세요.", e);
                } else if (cause instanceof java.util.concurrent.TimeoutException) {
                    throw new RuntimeException("Python 서버 응답 시간이 초과되었습니다.", e);
                } else {
                    throw new RuntimeException("Python 서버와 통신 실패: " + e.getMessage(), e);
                }
            }

            if (aiResponse == null) {
                throw new RuntimeException("Python 서버로부터 응답을 받지 못했습니다.");
            }
            
            System.out.println("✅ Python 서버 응답 수신: " + aiResponse.getAnswer().substring(0, Math.min(50, aiResponse.getAnswer().length())) + "...");

            // 4. [AI 답변] 저장은 Python 서버에서 이미 처리하므로 여기서는 저장하지 않음
            // (Python 서버의 chat_endpoint에서 save_to_firebase를 호출함)

            return aiResponse;
            
        } catch (Exception e) {
            System.err.println("❌ [ChatService] 처리 중 오류 발생: " + e.getMessage());
            e.printStackTrace();
            
            // 에러 응답 반환
            ChatResponse errorResponse = new ChatResponse();
            errorResponse.setAnswer("죄송합니다. 서버 오류가 발생했습니다: " + e.getMessage());
            errorResponse.setSources(java.util.Collections.emptyList());
            return errorResponse;
        }
    }

    // 파이어베이스 저장 도우미 함수
    private void saveMessageToFirebase(Firestore db, String roomName, String sender, String text) {
        try {
            Map<String, Object> message = new HashMap<>();
            message.put("sender", sender); // 누가 (user 또는 ai)
            message.put("message_type", sender); // Python 서버와 동일한 필드명 추가
            message.put("text", text);     // 내용
            message.put("timestamp", FieldValue.serverTimestamp()); // 서버 타임스탬프 사용

            // chat_rooms -> room_xxx -> messages -> 자동생성ID 문서에 저장
            db.collection("chat_rooms")
                    .document(roomName)
                    .collection("messages")
                    .add(message);
            
            System.out.println("🔥 Firebase 저장 완료: [" + sender + "] " + text);
        } catch (Exception e) {
            System.err.println("❌ Firebase 저장 실패: " + e.getMessage());
            e.printStackTrace();
        }
    }
}