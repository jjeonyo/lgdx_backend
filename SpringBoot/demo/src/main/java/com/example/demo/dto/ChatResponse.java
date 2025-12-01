package com.example.demo.dto;

import lombok.Getter;
import lombok.Setter;
import java.util.List;

@Getter @Setter
public class ChatResponse {
    private String answer;       // AI 답변
    private List<String> sources; // 참고한 매뉴얼 페이지
}