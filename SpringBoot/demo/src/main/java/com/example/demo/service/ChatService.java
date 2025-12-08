package com.example.demo.service;

import com.example.demo.dto.ChatRequest;
import com.example.demo.dto.ChatResponse;
import com.example.demo.dto.PythonRequest;
import com.google.cloud.firestore.Firestore;
import com.google.firebase.FirebaseApp;
import com.google.firebase.cloud.FirestoreClient;
import lombok.RequiredArgsConstructor;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.reactive.function.client.WebClient;
import reactor.netty.http.client.HttpClient;
import org.springframework.http.client.reactive.ReactorClientHttpConnector;
import java.time.Duration;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

import java.util.HashMap;
import java.util.Map;
import java.util.regex.Pattern;
import java.util.regex.Matcher;

@Service
@RequiredArgsConstructor
public class ChatService {

    @Value("${python.server.url:http://localhost:8000}")
    private String pythonServerUrl;

    private WebClient getWebClient() {
        HttpClient httpClient = HttpClient.create()
                .responseTimeout(Duration.ofSeconds(30)); // 응답 타임아웃 30초
        
        return WebClient.builder()
                .baseUrl(pythonServerUrl)
                .clientConnector(new ReactorClientHttpConnector(httpClient))
                .build();
    }

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
            
            // 방 이름 결정: sessionId가 있으면 사용, 없으면 기존 로직 사용
            String roomName;
            if (request.getSessionId() != null && !request.getSessionId().trim().isEmpty()) {
                roomName = request.getSessionId(); // 프론트엔드에서 전달된 room_id 사용 (예: room_user_001, room_user_002)
                System.out.println("📁 채팅방 (sessionId 사용): " + roomName);
            } else {
                roomName = "room_" + request.getUserId(); // 기존 로직 (하위 호환성)
                System.out.println("📁 채팅방 (기본값): " + roomName);
            }

            // 2. [사용자 질문] 저장은 Python 서버에서 처리하므로 여기서는 저장하지 않음
            // (Python 서버의 chat_endpoint에서 save_to_firebase를 호출함)

            // 3. 파이썬(AI)에게 질문하기
            PythonRequest pythonReq = new PythonRequest(request.getUserId(), request.getMessage(), roomName);
            System.out.println("📤 Python 서버로 요청 전송: " + pythonServerUrl + "/chat");
            System.out.println("📤 요청 내용 - userId: " + request.getUserId() + ", sessionId: " + roomName + ", message: " + request.getMessage().substring(0, Math.min(50, request.getMessage().length())) + "...");
            
            ChatResponse aiResponse;
            try {
                System.out.println("🔄 [Spring Boot] Python 서버 요청 시작...");
                aiResponse = getWebClient().post()
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
                        .doOnNext(response -> {
                            System.out.println("✅ [Spring Boot] Python 서버 응답 수신 성공!");
                            System.out.println("   - answer 길이: " + (response.getAnswer() != null ? response.getAnswer().length() : 0));
                            System.out.println("   - sources 개수: " + (response.getSources() != null ? response.getSources().size() : 0));
                        })
                        .doOnError(error -> {
                            System.err.println("❌ [Spring Boot] Python 서버 연결 실패: " + error.getClass().getSimpleName() + " - " + error.getMessage());
                            if (error.getCause() != null) {
                                System.err.println("   원인: " + error.getCause().getMessage());
                            }
                            error.printStackTrace();
                        })
                        .block();
                System.out.println("🔄 [Spring Boot] Python 서버 응답 대기 완료");
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

            // 4. [AI 답변] Firebase에 저장은 Python 서버에서 이미 처리하므로 여기서는 저장하지 않음
            // (Python 서버의 chat_endpoint에서 save_to_firebase를 호출함)
            System.out.println("✅ AI 답변은 Python 서버에서 이미 Firebase에 저장되었으므로 저장 생략");

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
            message.put("message_type", "chat"); // 메시지 타입: 'chat' (텍스트 챗봇)
            message.put("text", text);     // 내용
            // 타임스탬프 형식: "2025-12-05 14:38:02"
            message.put("timestamp", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));

            // chat_rooms -> room_xxx -> messages -> 자동생성ID 문서에 저장
            db.collection("chat_rooms")
                    .document(roomName)
                    .collection("messages")
                    .add(message);
            
            System.out.println("🔥 Firebase 저장 완료: [" + sender + "] " + text.substring(0, Math.min(50, text.length())) + "...");
        } catch (Exception e) {
            System.err.println("❌ Firebase 저장 실패: " + e.getMessage());
            e.printStackTrace();
        }
    }

    // 채팅방 삭제 및 새 room 생성 (room+1)
    public String deleteRoomAndCreateNew(String userId, String roomId) {
        try {
            System.out.println("🗑️ [ChatService] 채팅방 삭제 및 새 room 생성 시작 - userId: " + userId + ", roomId: " + roomId);
            
            // 1. Firebase 초기화 확인
            if (FirebaseApp.getApps().isEmpty()) {
                throw new RuntimeException("Firebase가 초기화되지 않았습니다.");
            }
            
            // 2. Firestore 가져오기
            Firestore db = FirestoreClient.getFirestore();
            if (db == null) {
                throw new RuntimeException("Firestore 연결 실패");
            }
            
            // 3. 기존 room_user_XXX 형태의 모든 room 조회
            var roomsRef = db.collection("chat_rooms");
            var roomsSnapshot = roomsRef.get().get();
            
            System.out.println("📋 [ChatService] 전체 rooms 조회 완료: " + roomsSnapshot.size() + "개");
            
            // 4. room_user_로 시작하는 문서들 중에서 가장 큰 숫자 찾기
            int maxNumber = 1;
            Pattern pattern = Pattern.compile("^room_user_(\\d+)$");
            
            for (var doc : roomsSnapshot.getDocuments()) {
                String docId = doc.getId();
                Matcher matcher = pattern.matcher(docId);
                if (matcher.matches()) {
                    try {
                        int number = Integer.parseInt(matcher.group(1));
                        if (number > maxNumber) {
                            maxNumber = number;
                        }
                        System.out.println("📋 [ChatService] room 발견: " + docId + " (숫자: " + number + ")");
                    } catch (NumberFormatException e) {
                        System.out.println("⚠️ [ChatService] 숫자 파싱 실패: " + docId);
                    }
                }
            }
            
            // 5. 새로운 room_id 생성 (가장 큰 숫자 + 1)
            int newRoomNumber = maxNumber + 1;
            String newRoomId = String.format("room_user_%03d", newRoomNumber); // 001, 002 형식
            
            System.out.println("✅ [ChatService] 새 room_id 생성: " + newRoomId + " (이전 최대값: " + maxNumber + ")");
            
            // 6. 새로운 room 문서 생성 (messages 서브컬렉션은 자동으로 생성됨)
            Map<String, Object> newRoomData = new HashMap<>();
            newRoomData.put("createdAt", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            newRoomData.put("updatedAt", LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss")));
            newRoomData.put("userId", userId);
            
            roomsRef.document(newRoomId).set(newRoomData).get();
            
            System.out.println("✅ [ChatService] 새 room 문서 생성 완료: " + newRoomId);
            System.out.println("✅ [ChatService] Firebase 경로: chat_rooms/" + newRoomId);
            
            return newRoomId;
            
        } catch (Exception e) {
            System.err.println("❌ [ChatService] 채팅방 삭제 및 새 room 생성 실패: " + e.getMessage());
            e.printStackTrace();
            throw new RuntimeException("채팅방 삭제 및 새 room 생성 실패: " + e.getMessage(), e);
        }
    }
}