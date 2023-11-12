//
//  ScrollingTerminalViewCell.swift
//  stockInterface
//
//  Created by CHONK on 11/1/23.
//

import Cocoa

class ScrollingTerminalViewCell: NSTableCellView {
    
    weak var label: NSTextField!
    weak var backgroundView: NSView!
    
    override init(frame frameRect: NSRect) {
        super.init(frame: frameRect)
        setupViews()
    }
    
    required init?(coder: NSCoder) {
        fatalError("init(coder:) has not been implemented")
    }
    
    func configure(text: String, type: TerminalTextType) {
        label.stringValue = text
        label.textColor = .white
        switch type {
        case .userInput: 
            backgroundView.layer?.backgroundColor = .TerminalUserInput
        case .error:
            backgroundView.layer?.backgroundColor = .TerminalError
        default: backgroundView.layer?.backgroundColor = .black
        }
    }
    
    private func setupViews() {
        let backgroundView = NSView()
        backgroundView.wantsLayer = true
        backgroundView.layer?.cornerRadius = 4
        backgroundView.layer?.backgroundColor = .black
        backgroundView.translatesAutoresizingMaskIntoConstraints = false
        addSubview(backgroundView)
        NSLayoutConstraint.activate([
            backgroundView.topAnchor.constraint(equalTo: topAnchor, constant: 0),
            backgroundView.leadingAnchor.constraint(equalTo: leadingAnchor),
            backgroundView.trailingAnchor.constraint(equalTo: trailingAnchor),
            backgroundView.bottomAnchor.constraint(equalTo: bottomAnchor, constant: 0)
        ])
        self.backgroundView = backgroundView
        
        let label = NSTextField(labelWithString: "")
        label.cell?.wraps = true
        label.font = NSFont(name: "BrassMono-Bold", size: 16)
        self.addSubview(label)
        label.translatesAutoresizingMaskIntoConstraints = false
        NSLayoutConstraint.activate([
            label.leadingAnchor.constraint(equalTo: self.leadingAnchor, constant: 10),
            label.trailingAnchor.constraint(equalTo: self.trailingAnchor, constant: -10),
            label.centerYAnchor.constraint(equalTo: centerYAnchor, constant: 1.5)
        ])
        self.label = label
    }
}
