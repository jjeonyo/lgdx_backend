package com.example.demo.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct; // (자바 버전에 따라 javax 대신 jakarta일 수 있음)
import java.io.FileInputStream;
import java.io.InputStream;

@Configuration
public class FirebaseConfig {

    @PostConstruct
    public void init() {
        InputStream serviceAccount = null;
        try {
            System.out.println("============================================");
            System.out.println("🔥 [DEBUG] 파이어베이스 연결 시도 중...");
            
            // 1. 환경변수에서 경로 가져오기 (spring-dotenv가 .env 파일에서 FIREBASE_KEY_PATH를 로드함)
            // System.getProperty()도 확인 (application.properties에서 설정된 경우)
            String firebaseKeyPath = System.getenv("FIREBASE_KEY_PATH");
            if (firebaseKeyPath == null || firebaseKeyPath.trim().isEmpty()) {
                firebaseKeyPath = System.getProperty("FIREBASE_KEY_PATH");
            }
            if (firebaseKeyPath == null || firebaseKeyPath.trim().isEmpty()) {
                // 환경변수가 없으면 기본 절대 경로 사용
                firebaseKeyPath = "C:\\dxfirebasekey\\serviceAccountKey.json";
            }
            
            // 2. 절대 경로에서 파일 찾기 시도
            try {
                serviceAccount = new FileInputStream(firebaseKeyPath);
                System.out.println("✅ [성공] 절대 경로에서 키 파일을 찾았습니다: " + firebaseKeyPath);
            } catch (Exception e) {
                System.out.println("⚠️ 절대 경로에서 파일을 찾지 못했습니다: " + firebaseKeyPath);
                System.out.println("   -> resources 폴더에서 찾기를 시도합니다...");
                
                // 3. resources 폴더에서 파일 찾기 시도
                serviceAccount = getClass().getClassLoader().getResourceAsStream("serviceAccountKey.json");
                
                if (serviceAccount == null) {
                    System.err.println("❌ [치명적 오류] serviceAccountKey.json 파일을 찾을 수 없습니다!");
                    System.err.println("   -> 환경변수 FIREBASE_KEY_PATH: " + (System.getenv("FIREBASE_KEY_PATH") != null ? System.getenv("FIREBASE_KEY_PATH") : "설정되지 않음"));
                    System.err.println("   -> 절대 경로: " + firebaseKeyPath);
                    System.err.println("   -> resources 폴더: src/main/resources/serviceAccountKey.json");
                    System.err.println("   -> 세 경로 모두 확인해주세요.");
                    throw new RuntimeException("파이어베이스 키 파일 누락 - 환경변수, 절대 경로, resources 폴더 모두에서 찾을 수 없습니다.");
                } else {
                    System.out.println("✅ [성공] resources 폴더에서 키 파일을 찾았습니다!");
                }
            }

            // 4. Firebase 초기화 (이미 초기화되어 있지 않은 경우만)
            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseOptions options = FirebaseOptions.builder()
                        .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                        .build();
                FirebaseApp.initializeApp(options);
                System.out.println("🎉 [완료] 파이어베이스 초기화 성공!");
            } else {
                System.out.println("ℹ️ 파이어베이스가 이미 초기화되어 있습니다.");
            }

            System.out.println("============================================");

        } catch (Exception e) {
            System.err.println("❌ [에러 발생] " + e.getMessage());
            e.printStackTrace();
            // 초기화 실패 시 예외를 던져서 서버 시작을 막음
            throw new RuntimeException("Firebase 초기화 실패: " + e.getMessage(), e);
        } finally {
            // 5. 스트림 닫기 (리소스 누수 방지)
            if (serviceAccount != null) {
                try {
                    serviceAccount.close();
                } catch (Exception e) {
                    System.err.println("⚠️ 스트림 닫기 실패: " + e.getMessage());
                }
            }
        }
    }
}