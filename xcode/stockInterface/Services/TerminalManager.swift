//
//  TerminalManager.swift
//  stockInterface
//
//  Created by Jackson Tubbs on 11/11/23.
//

import Foundation

class TerminalManager {
    
    static let shared = TerminalManager()
    
    private var commands: [String: Command] = [:]
    
    private weak var currentTerminal: ScrollingTerminalView? {
        didSet {
            applyPendingCommands()
        }
    }
    
    var pendingCommands: [(String, TerminalTextType)] = []
    
    private init() {
        registerCommand(HelpCommand())
        registerCommand(ChartCommand())
        registerCommand(InputOutputGenerationCommand())
        registerCommand(GetAllTickersListCommand())
        registerCommand(ChartAllTimeCommand())
        registerCommand(DownloadEveryStockCommand())
        registerCommand(BulkIndicatorDataCommand())
        registerCommand(ChartLocalCommand())
        registerCommand(SaveCommand())
        registerCommand(GenerateEntireDatasetCommand())
        registerCommand(BacktestCurrentCommand())
        registerCommand(GenerateGeneralAggregateCommand())
//        registerCommand(SaveCommand())
//        registerCommand(LoadPlayblackCommand())
//        registerCommand(ChartRandomCommand())
    }

    func registerCommand(_ command: Command) {
        commands[command.name] = command
    }
    
    func enterCommand(command: String) {
        currentTerminal?.addText(command, type: .userInput)
        let components = command.split(separator: " ").map(String.init)
        guard let commandName = components.first, let command = commands[commandName] else {
            currentTerminal?.addText("Invalid Command", type: .error)
            return
        }
        
        let arguments = Array(components.dropFirst())
        command.execute(with: arguments)
    }
    
    func addText(_ text: String, type: TerminalTextType = .normal) {
        if let terminal = self.currentTerminal {
            terminal.addText(text, type: type)
        } else {
            pendingCommands.append((text, type))
        }
    }
    
    func applyPendingCommands() {
        for e in pendingCommands {
            currentTerminal!.addText(e.0, type: e.1)
        }
        pendingCommands = []
    }
    
    func setTerminal(view: ScrollingTerminalView) {
        self.currentTerminal = view
    }
}
