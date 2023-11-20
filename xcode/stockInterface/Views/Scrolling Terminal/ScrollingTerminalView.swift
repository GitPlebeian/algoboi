//
//  ScrollingTerminalView.swift
//  stockInterface
//
//  Created by CHONK on 10/24/23.
//

import Cocoa

enum TerminalTextType {
    case userInput
    case normal
    case error
}

class ScrollingTerminalView: NSView {

    // MARK: Subviews
    
    weak var inputParentView: NSView!
    weak var inputField:      NSTextField!
    
    weak var scrollView:      NSScrollView!
    weak var tableView:       NSTableView!
    
    // MARK: Properties
    
    var cachedRowHeights: [CGFloat] = []
    var data: [String] = []
    var dataType: [TerminalTextType] = []
    
    
    // MARK: Style
    
    // MARK: Init
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        calculateCachedRowHeights()
        setupView()
        TerminalManager.shared.setTerminal(view: self)
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    // MARK: Public
    
    func addText(_ string: String, type: TerminalTextType = .normal) {
        data.append(string)
        cachedRowHeights.append(Text.HeightForString(string: string, width: 288 - 20, font: NSFont(name: "BrassMono-Bold", size: 16)!) + 5)
        dataType.append(type)
        tableView.reloadData()
        tableView.scrollRowToVisible(data.count - 1)
    }
    
    // MARK: Helpers
    
    private func calculateCachedRowHeights() {
        for e in data {
            cachedRowHeights.append(Text.HeightForString(string: e, width: 288 - 20, font: NSFont(name: "BrassMono-Bold", size: 16)!) + 5)
        }
    }
    
    // MARK: Setup View
    
    private func setupView() {
        wantsLayer = true
        layer?.backgroundColor = .black
        
        let inputParentView = NSView()
        inputParentView.wantsLayer = true
        inputParentView.layer?.backgroundColor = .InputBackground
        inputParentView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(inputParentView)
        NSLayoutConstraint.activate([
            inputParentView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 6),
            inputParentView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -6),
            inputParentView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: -6),
            inputParentView.heightAnchor.constraint(equalToConstant: 44)
        ])
        self.inputParentView = inputParentView
        
        let inputField = NSTextField()
        inputField.placeholderString = "Enter Command"
        inputField.font = NSFont(name: "BrassMono-Bold", size: 16)
        inputField.focusRingType = .none
        inputField.delegate = self
        inputField.resignFirstResponder()
        inputField.lineBreakMode = .byTruncatingTail
        inputField.isBordered = false
        inputField.backgroundColor = .clear
        inputField.translatesAutoresizingMaskIntoConstraints = false
        inputParentView.addSubview(inputField)
        NSLayoutConstraint.activate([
            inputField.leadingAnchor.constraint(equalTo: inputParentView.leadingAnchor, constant: 10),
            inputField.trailingAnchor.constraint(equalTo: inputParentView.trailingAnchor, constant: -10),
            inputField.centerYAnchor.constraint(equalTo: inputParentView.centerYAnchor)
        ])
        self.inputField = inputField
        
        let scrollView = NSScrollView()
        scrollView.hasVerticalScroller = true
        addSubview(scrollView)
        scrollView.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            scrollView.topAnchor.constraint(equalTo: topAnchor, constant: 0),
            scrollView.leadingAnchor.constraint(equalTo: leadingAnchor, constant: 0),
            scrollView.trailingAnchor.constraint(equalTo: trailingAnchor, constant: -0),
            scrollView.bottomAnchor.constraint(equalTo: inputParentView.topAnchor, constant: -6)
        ])
        self.scrollView = scrollView
        
        let tableView = NSTableView()
        tableView.delegate = self
        tableView.intercellSpacing = NSSize(width: 0, height: 3)
        tableView.style = .fullWidth
        tableView.backgroundColor = NSColor(cgColor: .InputBackground)!
        tableView.dataSource = self
        tableView.addTableColumn(NSTableColumn(identifier: NSUserInterfaceItemIdentifier(rawValue: "Column")))
        tableView.headerView = nil
        scrollView.documentView = tableView
        self.tableView = tableView
    }
}

extension ScrollingTerminalView: NSTextFieldDelegate {
    func control(_ control: NSControl, textView: NSTextView, doCommandBy commandSelector: Selector) -> Bool {
        
        if commandSelector == #selector(NSResponder.deleteBackward(_:)) {
            var text = inputField.stringValue
            text = String(text.dropLast())
            inputField.stringValue = text
        }
        
        if commandSelector == #selector(NSResponder.insertNewline(_:)) {
            TerminalManager.shared.enterCommand(command: inputField.stringValue)
            inputField.stringValue = ""
        }
        
        return true
    }
}

extension ScrollingTerminalView: NSTableViewDelegate, NSTableViewDataSource {
    func numberOfRows(in tableView: NSTableView) -> Int {
        return data.count
    }
    
    func tableView(_ tableView: NSTableView, viewFor tableColumn: NSTableColumn?, row: Int) -> NSView? {
        let cell = tableView.makeView(withIdentifier: NSUserInterfaceItemIdentifier(rawValue: "Cell"), owner: nil) as? ScrollingTerminalViewCell ?? ScrollingTerminalViewCell()
        cell.configure(text: data[row], type: dataType[row])
        return cell
    }
    
    func selectionShouldChange(in tableView: NSTableView) -> Bool {
        return false
    }
    
    func tableView(_ tableView: NSTableView, heightOfRow row: Int) -> CGFloat {
        return cachedRowHeights[row]
    }

}
