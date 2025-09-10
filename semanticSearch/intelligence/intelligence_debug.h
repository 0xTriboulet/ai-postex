#pragma once
#ifdef _DEBUG
#define PRINT(...) printf(__VA_ARGS__)
#define PAUSE() getchar();
#else
#define PRINT(...)
#define PAUSE() 
#endif