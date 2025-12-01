package com.example.demo.config;

import com.google.auth.oauth2.GoogleCredentials;
import com.google.firebase.FirebaseApp;
import com.google.firebase.FirebaseOptions;
import org.springframework.context.annotation.Configuration;

import jakarta.annotation.PostConstruct; // (자바 버전에 따라 javax 대신 jakarta일 수 있음)
import java.io.InputStream;

@Configuration
public class FirebaseConfig {

    @PostConstruct
    public void init() {
        try {
            System.out.println("============================================");
            System.out.println("🔥 [DEBUG] 파이어베이스 연결 시도 중...");
            
            // 1. 파일 읽기 시도
            InputStream serviceAccount = getClass().getClassLoader().getResourceAsStream("serviceAccountKey.json");

            // 2. 파일 있는지 검사 (여기가 핵심!)
            if (serviceAccount == null) {
                System.out.println("❌ [치명적 오류] serviceAccountKey.json 파일을 찾을 수 없습니다!");
                System.out.println("   -> src/main/resources 폴더에 파일이 있는지 다시 확인해주세요.");
                System.out.println("   -> 파일명에 오타나 띄어쓰기가 없는지 확인해주세요.");
                throw new RuntimeException("파이어베이스 키 파일 누락");
            } else {
                System.out.println("✅ [성공] 키 파일을 찾았습니다! 연결을 진행합니다.");
            }

            // 3. 연결
            if (FirebaseApp.getApps().isEmpty()) {
                FirebaseOptions options = FirebaseOptions.builder()
                        .setCredentials(GoogleCredentials.fromStream(serviceAccount))
                        .build();
                FirebaseApp.initializeApp(options);
                System.out.println("🎉 [완료] 파이어베이스 초기화 성공!");
            }

            System.out.println("============================================");

        } catch (Exception e) {
            System.out.println("❌ [에러 발생] " + e.getMessage());
            e.printStackTrace();
        }
    }
}