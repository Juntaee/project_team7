
// // 이전 대화까지 기억 성공 version_3
#if 00
#include <iostream>
#include <string>
#include "openai.hpp"
#include "nlohmann/json.hpp"

using namespace std;
using json = nlohmann::json;

int main() {
    openai::start();

    string previousUserContent = "";
    string previousGPTResponse = "";

    while (true) {
        cout << "Q : ";
        string userContent;
        cin.ignore();
        getline(cin, userContent);

        // 이전 대화와 현재 대화를 결합하여 JSON 객체 생성
        json messages = json::array();
        messages.push_back({ {"role", "user"}, {"content", previousUserContent} });
        messages.push_back({ {"role", "assistant"}, {"content", previousGPTResponse} });
        messages.push_back({ {"role", "user"}, {"content", userContent} });

        json jsonRequest = {
            {"model", "gpt-3.5-turbo"},
            {"messages", messages},
            {"max_tokens", 500},
            {"temperature", 0}
        };

        // Chat API 호출
        auto chat = openai::chat().create(jsonRequest);

        // GPT 응답 저장
        previousUserContent = userContent;
        previousGPTResponse = chat["choices"][0]["message"]["content"].get<string>();

        // 응답 출력
        cout << "GPT : " << previousGPTResponse << '\n';
    }

    return 0;
}
#endif

// 질문-답변 성공 version_2
#if 00
#include <iostream>
#include <string>
#include "openai.hpp"

using namespace std;

int main() {
    openai::start();

    while (1) {
        cout << "Q : ";
        string userContent;
        cin >> userContent;

        // JSON 문자열 생성
        string jsonString = R"(
        {
            "model": "gpt-3.5-turbo",
            "messages":[{"role":"user", "content":")" + userContent + R"("}],
            "max_tokens": 300,
            "temperature": 0
        }
        )";

        // JSON으로 변환
        nlohmann::json jsonRequest = nlohmann::json::parse(jsonString);

        // Chat API 호출
        auto chat = openai::chat().create(jsonRequest);

        // 응답 출력
        cout << "GPT : " << chat["choices"][0]["message"]["content"] << '\n';
    }

    return 0;
}
#endif




// 원본 예제 코드 version_1
#if 00
int main() {
    openai::start();

    auto chat = openai::chat().create(R"(
    {
        "model": "gpt-3.5-turbo",
        "messages":[{"role":"user", "content":"please recommend trevel spot in winter"}],
        "max_tokens": 7,
        "temperature": 0
    }
    )"_json);
    std::cout << "Response is:\n" << chat.dump(2) << '\n'; 
}
#endif
