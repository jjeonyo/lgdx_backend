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
        try {
            System.out.println("============================================");
            System.out.println("🔥 [DEBUG] 파이어베이스 연결 시도 중...");
            
            InputStream serviceAccount = null;
            
            // 1. 절대 경로에서 파일 찾기 시도 (Python 서버와 동일한 경로)
            String absolutePath = "C:\\dxfirebasekey\\serviceAccountKey.json";
            try {
                serviceAccount = new FileInputStream(absolutePath);
                System.out.println("✅ [성공] 절대 경로에서 키 파일을 찾았습니다: " + absolutePath);
            } catch (Exception e) {
                System.out.println("⚠️ 절대 경로에서 파일을 찾지 못했습니다: " + absolutePath);
                System.out.println("   -> resources 폴더에서 찾기를 시도합니다...");
                
                // 2. resources 폴더에서 파일 찾기 시도
                serviceAccount = getClass().getClassLoader().getResourceAsStream("serviceAccountKey.json");
                
                if (serviceAccount == null) {
                    System.err.println("❌ [치명적 오류] serviceAccountKey.json 파일을 찾을 수 없습니다!");
                    System.err.println("   -> 절대 경로: " + absolutePath);
                    System.err.println("   -> resources 폴더: src/main/resources/serviceAccountKey.json");
                    System.err.println("   -> 두 경로 모두 확인해주세요.");
                    throw new RuntimeException("파이어베이스 키 파일 누락 - 절대 경로와 resources 폴더 모두에서 찾을 수 없습니다.");
                } else {
                    System.out.println("✅ [성공] resources 폴더에서 키 파일을 찾았습니다!");
                }
            }

            // 3. Firebase 초기화 (이미 초기화되어 있지 않은 경우만)
            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseOptions options = FirebaseOptions.builder()
                        .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                        .build();
                FirebaseApp.initializeApp(options);
                System.out.println("🎉 [완료] 파이어베이스 초기화 성공!");
            } else {
                System.out.println("ℹ️ 파이어베이스가 이미 초기화되어 있습니다.");
            }
            
            // 4. 스트림 닫기
            if (serviceAccount != null) {
                serviceAccount.close();
            }

            System.out.println("============================================");

        } catch (Exception e) {
            System.err.println("❌ [에러 발생] " + e.getMessage());
            e.printStackTrace();
            // 초기화 실패 시 예외를 던져서 서버 시작을 막음
            throw new RuntimeException("Firebase 초기화 실패: " + e.getMessage(), e);
        }
    }
}