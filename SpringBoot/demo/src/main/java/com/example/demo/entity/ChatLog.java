package com.example.demo.entity;

import jakarta.persistence.*;
import lombok.Getter;
import lombok.NoArgsConstructor;
import lombok.Setter;
import java.time.LocalDateTime;

@Entity
@Getter @Setter
@NoArgsConstructor
public class ChatLog {

    @Id @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id; // 기록 번호

    private String userId; // 누가

    @Column(length = 1000)
    private String userQuestion; // 뭐라고 물어봤고

    @Column(length = 2000)
    private String aiAnswer; // AI가 뭐라고 답했는지

    private LocalDateTime createdAt; // 언제 (날짜시간)

    // 생성자 (기록할 때 편하게 쓰려고 만듦)
    public ChatLog(String userId, String userQuestion, String aiAnswer) {
        this.userId = userId;
        this.userQuestion = userQuestion;
        this.aiAnswer = aiAnswer;
        this.createdAt = LocalDateTime.now();
    }
}