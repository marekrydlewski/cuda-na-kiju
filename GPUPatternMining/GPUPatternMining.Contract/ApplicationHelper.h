#pragma once
#include <string>
#include <Windows.h>

class ApplicationHelper
{
public:
	static std::string getCurrentWorkingDirectory()
	{
		wchar_t buffer[MAX_PATH];
		char buf2[MAX_PATH];
		GetModuleFileName(NULL, buffer, MAX_PATH);
		wcstombs(buf2, buffer, MAX_PATH);
		std::string::size_type pos = std::string(buf2).find_last_of("\\/");
		return std::string(buf2).substr(0, pos);
	}
};