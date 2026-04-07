# Secure Environment Setup

이 프로젝트는 OpenAI API 키를 코드에 직접 넣지 않고 `.env` 파일 또는 운영체제 환경변수에서만 읽어야 합니다.

## .env 생성 방법

프로젝트 루트(`my3/`)에서 `.env.example`을 복사해 `.env`를 만듭니다.

```powershell
Copy-Item .env.example .env
```

생성한 `.env` 파일에서 `OPENAI_API_KEY` 값만 본인의 실제 키로 채웁니다. 실제 키는 절대 `.env.example`, 코드, 문서, 커밋 메시지에 넣지 마세요.

```powershell
notepad .env
```

예시 형식:

```dotenv
OPENAI_API_KEY=your_api_key_here
OPENAI_CHAT_MODEL=gpt-5.2
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072
```

PowerShell 세션에서만 임시로 환경변수를 설정하려면 아래처럼 실행할 수 있습니다.

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_CHAT_MODEL="gpt-5.2"
$env:OPENAI_EMBEDDING_MODEL="text-embedding-3-large"
$env:OPENAI_EMBEDDING_DIMENSIONS="3072"
```

## 읽는 환경변수

- `OPENAI_API_KEY`: OpenAI API 호출에 사용하는 키입니다.
- `OPENAI_CHAT_MODEL`: GPT 모델명입니다. 이 프로젝트에서는 `gpt-5.2`를 사용합니다.
- `OPENAI_EMBEDDING_MODEL`: 임베딩 모델명입니다. 이 프로젝트에서는 `text-embedding-3-large`를 사용합니다.
- `OPENAI_EMBEDDING_DIMENSIONS`: 임베딩 차원입니다. 이 프로젝트에서는 `3072`를 사용합니다.

## 보안 주의사항

API 키를 코드에 직접 넣으면 저장소 이력, 로그, 리뷰 도구, 공유 파일을 통해 외부에 노출될 수 있습니다. 반드시 `.env` 또는 환경변수로만 주입하고, 실제 키가 포함된 파일은 커밋하지 마세요.

## Git Commit 전 체크리스트

- `.env` 또는 `.env.*` 파일이 커밋 대상에 포함되지 않았는지 확인합니다.
- `.env.example`에는 실제 API 키가 없고 빈 placeholder만 있는지 확인합니다.
- 코드에 `OPENAI_API_KEY` 실제 값이 하드코딩되어 있지 않은지 확인합니다.
- 로컬 로그 파일(`*.log`, `logs/` 등)에 민감정보가 남아 커밋되지 않는지 확인합니다.
- `git status`와 `git diff --cached`로 스테이징된 파일을 확인합니다.
