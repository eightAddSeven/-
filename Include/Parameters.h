#ifndef PARAMETERS_H
#define PARAMETERS_H

#include <vector>
#include "ShortCut.h"
using namespace std;


class PluginParameters
{
public:
	static PluginParameters *GetInstance()
	{
		return pSelf;
	}
	void DestroyInstance(void);
	void SetIsNoPlugin(bool isNoPlugin) { noPlugin = isNoPlugin; };
	bool IsNoPlugins(void) {return noPlugin; };
	vector<PluginCmdShortcut> & GetPluginCommandList(void) {return pluginCommands; };
	vector<PluginCmdShortcut> & GetPluginCustomizedCmds(void) {return pluginCustomizedCmds; };
	const char *GetPluginPath(void) const {return nppPath; };
	const char *GetAppDataPath(void) const {return appdataNppDir; };
protected:
	PluginParameters();
	~PluginParameters(){};
	static PluginParameters *pSelf;
	vector<PluginCmdShortcut> pluginCommands;
	vector<PluginCmdShortcut>pluginCustomizedCmds;
	char shortcutsPath[MAX_PATH];
	char contextMenuPath[MAX_PATH];
	char nppPath[MAX_PATH];
	char appdataNppDir[MAX_PATH]; 
private:
	bool noPlugin;

};
#endif